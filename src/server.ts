import 'dotenv/config';
import express from 'express';
import Anthropic from '@anthropic-ai/sdk';

const app = express();
app.use(express.json());

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const GITHUB_TOKEN = process.env.GITHUB_TOKEN;
const GITHUB_REPO = process.env.GITHUB_REPO || 'takiguchi0706/company-context';
const GITHUB_BRANCH = process.env.GITHUB_BRANCH || 'master';

const tools: Anthropic.Tool[] = [
  {
    name: 'read_file',
    description: 'GitHubリポジトリからファイルを読み込む',
    input_schema: {
      type: 'object' as const,
      properties: {
        file_path: { type: 'string' }
      },
      required: ['file_path']
    }
  },
  {
    name: 'write_file',
    description: 'GitHubリポジトリにファイルを書き込む（コミット）',
    input_schema: {
      type: 'object' as const,
      properties: {
        file_path: { type: 'string' },
        content: { type: 'string' }
      },
      required: ['file_path', 'content']
    }
  },
  {
    name: 'list_files',
    description: 'GitHubリポジトリの指定ディレクトリのファイル一覧を取得する',
    input_schema: {
      type: 'object' as const,
      properties: {
        directory: { type: 'string', description: 'サブディレクトリ（省略時はルート）' }
      },
      required: []
    }
  }
];

async function githubGet(path: string): Promise<Response> {
  return fetch(`https://api.github.com/repos/${GITHUB_REPO}/contents/${path}?ref=${GITHUB_BRANCH}`, {
    headers: {
      Authorization: `Bearer ${GITHUB_TOKEN}`,
      Accept: 'application/vnd.github.v3+json',
    },
  });
}

async function githubPut(path: string, body: Record<string, string>): Promise<Response> {
  return fetch(`https://api.github.com/repos/${GITHUB_REPO}/contents/${path}`, {
    method: 'PUT',
    headers: {
      Authorization: `Bearer ${GITHUB_TOKEN}`,
      Accept: 'application/vnd.github.v3+json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
}

async function executeTool(name: string, input: Record<string, string>, department: string): Promise<string> {
  try {
    const filePath = `${department}/${input.file_path}`;

    if (name === 'read_file') {
      const res = await githubGet(filePath);
      if (!res.ok) {
        return `Error: ファイルが見つかりません (${filePath}): ${res.status} ${res.statusText}`;
      }
      const data = await res.json() as { content: string };
      return Buffer.from(data.content, 'base64').toString('utf-8');
    }

    if (name === 'write_file') {
      // 既存ファイルのSHAを取得（存在する場合）
      let sha: string | undefined;
      const getRes = await githubGet(filePath);
      if (getRes.ok) {
        const existing = await getRes.json() as { sha: string };
        sha = existing.sha;
      }

      const body: Record<string, string> = {
        message: `agent: update ${filePath}`,
        content: Buffer.from(input.content, 'utf-8').toString('base64'),
        branch: GITHUB_BRANCH,
      };
      if (sha) {
        body.sha = sha;
      }

      const putRes = await githubPut(filePath, body);
      if (!putRes.ok) {
        const errText = await putRes.text();
        return `Error: ファイルの書き込みに失敗しました (${filePath}): ${putRes.status} ${errText}`;
      }
      return `Written: ${filePath}`;
    }

    if (name === 'list_files') {
      const dir = input.directory
        ? `${department}/${input.directory}`
        : department;
      const res = await githubGet(dir);
      if (!res.ok) {
        return `Error: ディレクトリが見つかりません (${dir}): ${res.status} ${res.statusText}`;
      }
      const data = await res.json() as Array<{ name: string; type: string }>;
      const files = data.map(f => `${f.type === 'dir' ? '[dir]' : '[file]'} ${f.name}`).join('\n');
      return files || '(空のディレクトリ)';
    }

    return `Unknown tool: ${name}`;
  } catch (e) {
    return `Error: ${String(e)}`;
  }
}

const COST_PER_INPUT = 3 / 1_000_000;
const COST_PER_OUTPUT = 15 / 1_000_000;

type Usage = { input_tokens: number; output_tokens: number; cost_usd: number };

async function executeAgent(
  instruction: string,
  department: string,
  system_prompt?: string
): Promise<{ result: string; messages: Anthropic.MessageParam[]; usage: Usage }> {
  const messages: Anthropic.MessageParam[] = [
    { role: 'user', content: instruction }
  ];

  let finalText = '';
  let totalInputTokens = 0;
  let totalOutputTokens = 0;

  while (true) {
    const response = await client.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 8096,
      tools,
      system: system_prompt ?? undefined,
      messages,
    });

    totalInputTokens += response.usage.input_tokens;
    totalOutputTokens += response.usage.output_tokens;

    messages.push({ role: 'assistant', content: response.content });

    if (response.stop_reason === 'end_turn') {
      finalText = response.content
        .filter((b): b is Anthropic.TextBlock => b.type === 'text')
        .map(b => b.text)
        .join('\n');
      break;
    }

    if (response.stop_reason === 'tool_use') {
      const toolResults: Anthropic.ToolResultBlockParam[] = await Promise.all(
        response.content
          .filter((b): b is Anthropic.ToolUseBlock => b.type === 'tool_use')
          .map(async (block) => ({
            type: 'tool_result' as const,
            tool_use_id: block.id,
            content: await executeTool(block.name, block.input as Record<string, string>, department),
          }))
      );

      messages.push({ role: 'user', content: toolResults });
      continue;
    }

    break;
  }

  const cost_usd = totalInputTokens * COST_PER_INPUT + totalOutputTokens * COST_PER_OUTPUT;
  return { result: finalText, messages, usage: { input_tokens: totalInputTokens, output_tokens: totalOutputTokens, cost_usd } };
}

app.post('/execute-agent', async (req, res) => {
  const { instruction, department, system_prompt } = req.body;

  try {
    const { result, messages, usage } = await executeAgent(instruction, department, system_prompt);
    res.json({ success: true, result, messages, usage });
  } catch (error) {
    console.error(error);
    res.status(500).json({ success: false, error: String(error) });
  }
});

async function executeAgentStream(
  instruction: string,
  department: string,
  system_prompt: string | undefined,
  res: express.Response
): Promise<void> {
  const messages: Anthropic.MessageParam[] = [
    { role: 'user', content: instruction }
  ];

  let fullContent = '';
  let totalInputTokens = 0;
  let totalOutputTokens = 0;

  try {
    while (true) {
      const stream = client.messages.stream({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 8096,
        tools,
        system: system_prompt ?? undefined,
        messages,
      });

      for await (const event of stream) {
        if (event.type === 'message_start') {
          totalInputTokens += event.message.usage.input_tokens;
        } else if (event.type === 'message_delta') {
          totalOutputTokens += event.usage.output_tokens;
        } else if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
          fullContent += event.delta.text;
          res.write(`data: ${JSON.stringify({ type: 'content_delta', text: event.delta.text })}\n\n`);
        }
      }

      const finalMessage = await stream.finalMessage();
      messages.push({ role: 'assistant', content: finalMessage.content });

      if (finalMessage.stop_reason === 'end_turn') break;

      if (finalMessage.stop_reason === 'tool_use') {
        const toolResults: Anthropic.ToolResultBlockParam[] = await Promise.all(
          finalMessage.content
            .filter((b): b is Anthropic.ToolUseBlock => b.type === 'tool_use')
            .map(async (block) => ({
              type: 'tool_result' as const,
              tool_use_id: block.id,
              content: await executeTool(block.name, block.input as Record<string, string>, department),
            }))
        );
        messages.push({ role: 'user', content: toolResults });
        continue;
      }

      break;
    }

    const cost_usd = totalInputTokens * COST_PER_INPUT + totalOutputTokens * COST_PER_OUTPUT;
    res.write(`event: done\ndata: ${JSON.stringify({
      usage: { input_tokens: totalInputTokens, output_tokens: totalOutputTokens, cost_usd },
      content: fullContent,
    })}\n\n`);
    res.end();
  } catch (error) {
    res.write(`event: error\ndata: ${JSON.stringify({ error: String(error) })}\n\n`);
    res.end();
  }
}

app.post('/execute-agent-stream', async (req, res) => {
  const { instruction, department, system_prompt } = req.body;

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  await executeAgentStream(instruction, department, system_prompt, res);
});

app.post('/execute-bulk', async (req, res) => {
  const { instruction, departments, system_prompts } = req.body as {
    instruction: string;
    departments: string[];
    system_prompts?: Record<string, string>;
  };

  if (!instruction || !departments || !Array.isArray(departments) || departments.length === 0) {
    res.status(400).json({ success: false, error: 'instruction and departments are required' });
    return;
  }

  const results = await Promise.all(
    departments.map(async (dept) => {
      try {
        const systemPrompt = system_prompts?.[dept];
        const { result, usage } = await executeAgent(instruction, dept, systemPrompt);
        return { department: dept, success: true, result, usage };
      } catch (error) {
        return { department: dept, success: false, error: String(error) };
      }
    })
  );

  const totalUsage = results.reduce(
    (acc, r) => {
      if (r.success && 'usage' in r && r.usage) {
        acc.input_tokens += r.usage.input_tokens;
        acc.output_tokens += r.usage.output_tokens;
        acc.cost_usd += r.usage.cost_usd;
      }
      return acc;
    },
    { input_tokens: 0, output_tokens: 0, cost_usd: 0 }
  );

  res.json({ success: true, results, usage: totalUsage });
});

// SSEストリーミング版: 部署ごとに完了次第 data イベントを送信
app.post('/execute-bulk-stream', async (req, res) => {
  const { instruction, departments, system_prompts } = req.body as {
    instruction: string;
    departments: string[];
    system_prompts?: Record<string, string>;
  };

  if (!instruction || !departments || !Array.isArray(departments) || departments.length === 0) {
    res.status(400).json({ success: false, error: 'instruction and departments are required' });
    return;
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  // 部署を直列で実行し、完了次第ストリームに書き出す
  for (const dept of departments) {
    try {
      const systemPrompt = system_prompts?.[dept];
      const { result, usage } = await executeAgent(instruction, dept, systemPrompt);
      const payload = JSON.stringify({ department: dept, success: true, result, usage });
      res.write(`data: ${payload}\n\n`);
    } catch (error) {
      const payload = JSON.stringify({ department: dept, success: false, error: String(error) });
      res.write(`data: ${payload}\n\n`);
    }
  }

  res.write('event: done\ndata: {}\n\n');
  res.end();
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Agent server running on port ${PORT}`);
});
