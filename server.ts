import express from 'express';
import path from 'path';
import fs from 'fs';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';
import { ZipArchive } from 'archiver';

dotenv.config();

const IS_PROD = process.env.NODE_ENV === 'production';
const PORT = process.env.PORT || 3000;

async function startServer() {
  const app = express();

  app.use(express.json({ limit: '5mb' }));

  // Shared lazy Gemini client
  let genAIClient: GoogleGenAI | null = null;
  function getGeminiClient(): GoogleGenAI {
    if (!genAIClient) {
      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey) {
        throw new Error('GEMINI_API_KEY is not configured in the environment.');
      }
      genAIClient = new GoogleGenAI({
        apiKey,
        httpOptions: {
          headers: {
            'User-Agent': 'aistudio-build',
          },
        },
      });
    }
    return genAIClient;
  }

  // In-memory Gemini Admin & Cost Metrics Store
  const adminMetrics = {
    totalRequests: 0,
    successfulRequests: 0,
    failedRequests: 0,
    totalPromptTokens: 0,
    totalCompletionTokens: 0,
    totalTokens: 0,
    totalEstimatedCostUsd: 0,
    queryTypeBreakdown: {
      standard: 0,
      deep_tafsir: 0,
      scholarly_analysis: 0,
    },
    modelBreakdown: {} as Record<string, number>,
    totalLatencyMs: 0,
    recentLogs: [] as Array<{
      id: string;
      timestamp: string;
      queryType: string;
      model: string;
      promptTokens: number;
      completionTokens: number;
      totalTokens: number;
      estimatedCostUsd: number;
      durationMs: number;
      status: 'success' | 'error';
      language: string;
      snippet: string;
    }>,
    serverStartTime: new Date().toISOString(),
  };

  const ISLAMIC_SCHOLAR_SYSTEM_INSTRUCTION = `You are a knowledgeable, wise, and moderate Islamic AI scholar and Quranic interpreter for the "Al-Iman" app.

CORE PRINCIPLES:
1. ACCURACY: Base all answers strictly on the Holy Quran, authentic Sunnah, and recognized classical Tafsir (e.g., Ibn Kathir, Al-Tabari, Al-Qurtubi, Al-Sa'di).
2. PERSONALITY: Adopt a gentle, respectful, calm, and reassuring tone (Wasatiyyah). Address the user as "Brother/Sister" or "Dear seeker of knowledge".
3. NO FABRICATION: Never invent Hadiths or provide uncertain information. If a Hadith is weak or disputed, mention it or avoid using it as a primary evidence.
4. JURISPRUDENCE: When discussing differences (Fiqh), clearly explain that there are multiple valid scholarly opinions across the 4 major Madhabs (Hanafi, Maliki, Shafi'i, Hanbali) with respect.
5. DISCLAIMER: For complex legal Fatwas (e.g., divorce, complex inheritance, judicial matters), gently advise consulting a certified local Mufti or official Islamic authority.

FORMATTING RULES (IMPORTANT for UI):
- Use clear Markdown headings (###) for structure.
- Use bullet points for lists.
- Use bold (**text**) for Surah names, key terms, and rulings.
- Use blockquotes (>) for Quranic verses and Hadiths. Always include citations (Surah:Ayah, Collection:HadithNo).
- Divide long answers into logical paragraphs.
- Keep the language clean and easy to read on mobile devices.

LANGUAGE:
- Always reply in the same language as the user's query (Arabic, English, or French).
- If the app context specifies a language, strictly follow it.`;

  // API endpoint for Islamic Scholar Chat with Query Type and Credits Depth
  app.post('/api/scholar/chat', async (req, res) => {
    const startTime = Date.now();
    const { messages, language = 'ar', queryType = 'standard' } = req.body;

    const safeQueryType: 'standard' | 'deep_tafsir' | 'scholarly_analysis' =
      queryType === 'deep_tafsir' || queryType === 'scholarly_analysis' ? queryType : 'standard';

    adminMetrics.totalRequests += 1;
    adminMetrics.queryTypeBreakdown[safeQueryType] = (adminMetrics.queryTypeBreakdown[safeQueryType] || 0) + 1;

    try {
      if (!messages || !Array.isArray(messages) || messages.length === 0) {
        adminMetrics.failedRequests += 1;
        return res.status(400).json({ error: 'Messages array is required' });
      }

      const ai = getGeminiClient();

      const langInstruction =
        language === 'fr'
          ? 'CURRENT SESSION LANGUAGE: French (Français). You MUST reply completely in French, providing French explanations with authentic Arabic citations and French translations.'
          : language === 'en'
          ? 'CURRENT SESSION LANGUAGE: English. You MUST reply completely in English, providing English explanations with authentic Arabic citations and English translations.'
          : 'CURRENT SESSION LANGUAGE: Arabic (العربية). You MUST reply completely in eloquent, clear Arabic with authentic citations.';

      // Tailor system instructions based on depth mode
      let depthInstruction = '';
      if (safeQueryType === 'deep_tafsir') {
        depthInstruction = `\n\nMODE: DEEP QURANIC TAFSIR (التفسير العميق والبياني للآيات والسور):
- Provide an exhaustive, profound exegesis of the cited verses.
- Break down key Arabic linguistic roots (المفردات البلاغية واللغوية).
- Detail the authentic causes and context of revelation (أسباب النزول وسياق الآيات).
- Synthesize interpretations from classical exegetes (تفسير ابن كثير، القرطبي، الطبري، والسعدي).
- Highlight spiritual reflections and practical life lessons (الهدايات والفوائد الإيمانية والعملية).`;
      } else if (safeQueryType === 'scholarly_analysis') {
        depthInstruction = `\n\nMODE: ADVANCED COMPARATIVE JURISPRUDENCE (التحليل الفقهي المقارن وأقوال العلماء والمذاهب):
- Provide a structured comparative legal analysis across the 4 Sunni Madhabs (Hanafi, Maliki, Shafi'i, Hanbali).
- Cite the foundational textual evidence (الأدلة من القرآن وصحيح السنة وقواعد أصول الفقه) for each viewpoint.
- Mention the consensus (الإجماع) or reasons for legitimate scholarly divergence (سبب اختلاف الفقهاء).
- Cite rulings and recommendations from recognized contemporary international Islamic Fiqh academies where relevant.`;
      } else {
        depthInstruction = `\n\nMODE: STANDARD CONCISE GUIDANCE (السؤال الديني المباشر):
- Deliver clear, direct, and well-structured guidance with authentic citations.`;
      }

      const effectiveSystemInstruction = `${ISLAMIC_SCHOLAR_SYSTEM_INSTRUCTION}\n\n${langInstruction}${depthInstruction}`;

      // Transform messages into Gemini contents format
      const historyContents = messages
        .filter((m: any, idx: number) => {
          if (idx === 0 && (m.role === 'assistant' || m.role === 'model')) {
            return false;
          }
          return Boolean(m.content && m.content.trim());
        })
        .map((m: any) => ({
          role: m.role === 'assistant' || m.role === 'model' ? 'model' : 'user',
          parts: [{ text: m.content }],
        }));

      if (historyContents.length === 0) {
        adminMetrics.failedRequests += 1;
        return res.status(400).json({ error: 'No valid user messages provided' });
      }

      const lastUserMsg = messages[messages.length - 1]?.content || '';
      const snippet = lastUserMsg.slice(0, 80);

      // Comprehensive list of active Gemini models with instant fallback
      const CANDIDATE_MODELS = [
        'gemini-3.1-flash-lite',
        'gemini-flash-latest',
        'gemini-3.6-flash',
        'gemini-3.5-flash',
        'gemini-flash-lite-latest',
        'gemini-3.7-flash',
      ];

      let lastError: any = null;
      let replyText = '';
      let successfulModel = '';

      for (const modelName of CANDIDATE_MODELS) {
        try {
          const response = await ai.models.generateContent({
            model: modelName,
            contents: historyContents,
            config: {
              systemInstruction: effectiveSystemInstruction,
              temperature: safeQueryType === 'deep_tafsir' ? 0.4 : safeQueryType === 'scholarly_analysis' ? 0.3 : 0.6,
            },
          });

          replyText = response.text || '';
          if (replyText && replyText.trim()) {
            successfulModel = modelName;
            break;
          }
        } catch (err: any) {
          lastError = err;
          const errMsg = String(err?.message || err || '');
          console.info(`[Islamic Scholar API] Model "${modelName}" note: ${errMsg.slice(0, 80)}, switching to fallback model...`);
          continue;
        }
      }

      if (!replyText && lastError) {
        throw lastError;
      }

      const durationMs = Date.now() - startTime;
      const finalReply = replyText.trim() || 'الحمد لله رب العالمين.';

      // Token estimation: approx 1 token per 3.5 chars
      const estimatedPromptTokens = Math.max(20, Math.round(JSON.stringify(historyContents).length / 3.5));
      const estimatedCompletionTokens = Math.max(20, Math.round(finalReply.length / 3.5));
      const totalTokens = estimatedPromptTokens + estimatedCompletionTokens;

      // Gemini Flash pricing model: ~$0.075 / 1M prompt, $0.30 / 1M completion
      const estimatedCost = (estimatedPromptTokens * 0.000000075) + (estimatedCompletionTokens * 0.00000030);

      // Update metrics
      adminMetrics.successfulRequests += 1;
      adminMetrics.totalPromptTokens += estimatedPromptTokens;
      adminMetrics.totalCompletionTokens += estimatedCompletionTokens;
      adminMetrics.totalTokens += totalTokens;
      adminMetrics.totalEstimatedCostUsd += estimatedCost;
      adminMetrics.totalLatencyMs += durationMs;
      adminMetrics.modelBreakdown[successfulModel || 'gemini-flash'] = (adminMetrics.modelBreakdown[successfulModel || 'gemini-flash'] || 0) + 1;

      // Add to recent logs (keep last 30)
      adminMetrics.recentLogs.unshift({
        id: `req_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
        timestamp: new Date().toISOString(),
        queryType: safeQueryType,
        model: successfulModel || 'gemini-flash',
        promptTokens: estimatedPromptTokens,
        completionTokens: estimatedCompletionTokens,
        totalTokens,
        estimatedCostUsd: Number(estimatedCost.toFixed(6)),
        durationMs,
        status: 'success',
        language,
        snippet,
      });
      if (adminMetrics.recentLogs.length > 30) {
        adminMetrics.recentLogs.pop();
      }

      return res.json({
        reply: finalReply,
        queryType: safeQueryType,
        modelUsed: successfulModel,
        tokens: {
          prompt: estimatedPromptTokens,
          completion: estimatedCompletionTokens,
          total: totalTokens,
        },
      });
    } catch (error: any) {
      const durationMs = Date.now() - startTime;
      adminMetrics.failedRequests += 1;
      console.error('[Islamic Scholar API Error]:', error);

      adminMetrics.recentLogs.unshift({
        id: `req_err_${Date.now()}`,
        timestamp: new Date().toISOString(),
        queryType: safeQueryType,
        model: 'failed',
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
        estimatedCostUsd: 0,
        durationMs,
        status: 'error',
        language,
        snippet: (req.body?.messages?.[req.body?.messages?.length - 1]?.content || '').slice(0, 80),
      });

      const isApiKeyMissing = !process.env.GEMINI_API_KEY;
      const isCapacityError =
        error?.status === 503 ||
        error?.message?.includes('503') ||
        error?.message?.includes('high demand') ||
        error?.message?.includes('UNAVAILABLE') ||
        error?.status === 429;

      const lang = req.body?.language || 'ar';
      let userFacingError =
        lang === 'fr'
          ? 'Une erreur est survenue lors du traitement de votre demande. Veuillez réessayer.'
          : lang === 'en'
          ? 'An error occurred while processing your request. Please try again.'
          : 'حدث خطأ أثناء معالجة الطلب، يرجى المحاولة مرة أخرى.';

      if (isApiKeyMissing) {
        userFacingError =
          lang === 'fr'
            ? 'La clé API Gemini n’est pas configurée.'
            : lang === 'en'
            ? 'Gemini API key is not configured on the server.'
            : 'مفتاح Gemini API غير مهيأ في الخادم.';
      } else if (isCapacityError) {
        userFacingError =
          lang === 'fr'
            ? 'Le service fait face à une forte demande temporaire. Veuillez réessayer dans quelques instants.'
            : lang === 'en'
            ? 'High traffic temporarily. Please click retry in a few seconds.'
            : 'الخدمة تشهد ضغطاً مؤقتاً في الطلبات، يرجى الضغط على إعادة المحاولة بعد بضع ثوانٍ.';
      }

      return res.status(500).json({
        error: userFacingError,
      });
    }
  });

  // Admin Metrics API for Gemini API and Cost Monitoring Dashboard
  app.get('/api/admin/metrics', (req, res) => {
    const avgLatency =
      adminMetrics.successfulRequests > 0
        ? Math.round(adminMetrics.totalLatencyMs / adminMetrics.successfulRequests)
        : 0;

    res.json({
      totalRequests: adminMetrics.totalRequests,
      successfulRequests: adminMetrics.successfulRequests,
      failedRequests: adminMetrics.failedRequests,
      totalPromptTokens: adminMetrics.totalPromptTokens,
      totalCompletionTokens: adminMetrics.totalCompletionTokens,
      totalTokens: adminMetrics.totalTokens,
      totalEstimatedCostUsd: Number(adminMetrics.totalEstimatedCostUsd.toFixed(6)),
      queryTypeBreakdown: adminMetrics.queryTypeBreakdown,
      modelBreakdown: adminMetrics.modelBreakdown,
      averageLatencyMs: avgLatency,
      recentLogs: adminMetrics.recentLogs,
      serverStartTime: adminMetrics.serverStartTime,
    });
  });

  // Admin Reset Metrics
  app.post('/api/admin/reset-metrics', (req, res) => {
    adminMetrics.totalRequests = 0;
    adminMetrics.successfulRequests = 0;
    adminMetrics.failedRequests = 0;
    adminMetrics.totalPromptTokens = 0;
    adminMetrics.totalCompletionTokens = 0;
    adminMetrics.totalTokens = 0;
    adminMetrics.totalEstimatedCostUsd = 0;
    adminMetrics.totalLatencyMs = 0;
    adminMetrics.queryTypeBreakdown = { standard: 0, deep_tafsir: 0, scholarly_analysis: 0 };
    adminMetrics.modelBreakdown = {};
    adminMetrics.recentLogs = [];
    res.json({ success: true, message: 'Metrics reset successfully' });
  });

  // Google TTS Proxy Endpoint (bypasses CORS on PC/Web browser for crystal clear Arabic voice)
  app.get('/api/tts', async (req, res) => {
    try {
      const text = req.query.text as string;
      if (!text || !text.trim()) {
        return res.status(400).send('No text provided');
      }

      const encodedText = encodeURIComponent(text.slice(0, 200));
      const googleTtsUrl = `https://translate.google.com/translate_tts?ie=UTF-8&q=${encodedText}&tl=ar&client=tw-ob`;

      const response = await fetch(googleTtsUrl, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
          'Referer': 'https://translate.google.com/',
        },
      });

      if (!response.ok) {
        return res.status(500).send('Google TTS returned error');
      }

      const buffer = await response.arrayBuffer();
      res.setHeader('Content-Type', 'audio/mpeg');
      res.setHeader('Cache-Control', 'public, max-age=86400');
      res.send(Buffer.from(buffer));
    } catch (e) {
      console.error('[TTS Proxy Error]:', e);
      res.status(500).send('TTS Proxy error');
    }
  });

  // Health check endpoint
  app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', time: new Date().toISOString() });
  });

  // Direct APK Download Endpoint
  app.get('/download-apk', (req, res) => {
    const apkPath = path.join('C:', 'Users', 'janat', 'Downloads', 'issam11111', 'android', 'app', 'build', 'outputs', 'apk', 'debug', 'app-debug.apk');
    if (fs.existsSync(apkPath)) {
      res.setHeader('Content-Type', 'application/vnd.android.package-archive');
      res.setHeader('Content-Disposition', 'attachment; filename="Al-Iman-App.apk"');
      const fileStream = fs.createReadStream(apkPath);
      fileStream.pipe(res);
    } else {
      res.status(404).send('APK file not found. Please build the project first.');
    }
  });

  // Direct Project ZIP Source Code Export Endpoint
  app.get('/api/export-zip', (req, res) => {
    try {
      const archive = new ZipArchive({
        zlib: { level: 9 }, // Best compression
      });

      res.setHeader('Content-Type', 'application/zip');
      res.setHeader(
        'Content-Disposition',
        'attachment; filename="Islamic-Prayer-Times-AI-SourceCode.zip"'
      );

      archive.on('error', (err) => {
        console.error('Archive error:', err);
        if (!res.headersSent) {
          res.status(500).send('Error creating ZIP archive');
        }
      });

      archive.pipe(res);

      const rootDir = process.cwd();

      // Directories to include
      const dirsToInclude = ['android', 'dist', 'src', 'public', 'assets'];
      for (const dir of dirsToInclude) {
        const fullPath = path.join(rootDir, dir);
        if (fs.existsSync(fullPath)) {
          archive.directory(fullPath, dir);
        }
      }

      // Root files to include
      const filesToInclude = [
        'package.json',
        'capacitor.config.ts',
        'tsconfig.json',
        'vite.config.ts',
        'server.ts',
        'index.html',
        'metadata.json',
        '.env.example',
        '.gitignore',
      ];

      for (const file of filesToInclude) {
        const fullPath = path.join(rootDir, file);
        if (fs.existsSync(fullPath)) {
          archive.file(fullPath, { name: file });
        }
      }

      archive.finalize();
    } catch (error) {
      console.error('Export ZIP error:', error);
      if (!res.headersSent) {
        res.status(500).send('Failed to generate ZIP');
      }
    }
  });

  // Vite middleware in dev or static files in production
  if (!IS_PROD) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[Islamic App Server] Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
