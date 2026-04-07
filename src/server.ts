import 'dotenv/config';
import express from 'express';
import Anthropic from '@anthropic-ai/sdk';
import * as fs from 'fs';
import * as path from 'path';

const app = express();
app.use(express.json());

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const COMPANY_ROOT = 'C:\\Users\\valuc\\projects\\company';

const DEPARTMENT_PATHS: Record<string, string> = {
  '01_secretary': '01_secretary',
  '02_planning': '02_planning',
  '03_design': '03_design',
  '04_product': '04_product',
  '05_content': '05_content',
  '06_automation': '06_automation',
  '07_finance': '07_finance',
  '08_marketing': '08_marketing',
  '09_analytics': '09_analytics',
  '10_legal': '10_legal',
  '11_qa': '11_qa',
};

function getDepartmentPath(department: string): string {
  const subPath = DEPARTMENT_PATHS[department];
  if (!subPath) throw new Error(`Unknown department: ${department}`);
  return path.join(COMPANY_ROOT, subPath);
}

const tools: Anthropic.Tool[] = [
  {
    name: 'read_file',
    description: 'ファイルを読み込む',
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
    description: 'ファイルを書き込む',
    input_schema: {
      type: 'object' as const,
      properties: {
        file_path: { type: 'string' },
        content: { type: 'string' }
      },
      required: ['file_path', 'content']
    }
  }
];

function executeTool(name: string, input: Record<string, string>, department: string): string {
  try {
    const basePath = getDepartmentPath(department);
    if (name === 'read_file') {
      const fullPath = path.resolve(basePath, input.file_path);
      return fs.readFileSync(fullPath, 'utf-8');
    }
    if (name === 'write_file') {
      const fullPath = path.resolve(basePath, input.file_path);
      fs.mkdirSync(path.dirname(fullPath), { recursive: true });
      fs.writeFileSync(fullPath, input.content, 'utf-8');
      return `Written: ${fullPath}`;
    }
    return `Unknown tool: ${name}`;
  } catch (e) {
    return `Error: ${String(e)}`;
  }
}

app.post('/execute-agent', async (req, res) => {
  const { instruction, department, system_prompt } = req.body;

  try {
    const messages: Anthropic.MessageParam[] = [
      { role: 'user', content: instruction }
    ];

    let finalText = '';

    while (true) {
      const response = await client.messages.create({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 8096,
        tools,
        system: system_prompt ?? undefined,
        messages,
      });

      messages.push({ role: 'assistant', content: response.content });

      if (response.stop_reason === 'end_turn') {
        finalText = response.content
          .filter((b): b is Anthropic.TextBlock => b.type === 'text')
          .map(b => b.text)
          .join('\n');
        break;
      }

      if (response.stop_reason === 'tool_use') {
        const toolResults: Anthropic.ToolResultBlockParam[] = response.content
          .filter((b): b is Anthropic.ToolUseBlock => b.type === 'tool_use')
          .map(block => ({
            type: 'tool_result' as const,
            tool_use_id: block.id,
            content: executeTool(block.name, block.input as Record<string, string>, department),
          }));

        messages.push({ role: 'user', content: toolResults });
        continue;
      }

      break;
    }

    res.json({ success: true, result: finalText, messages });
  } catch (error) {
    console.error(error);
    res.status(500).json({ success: false, error: String(error) });
  }
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Agent server running on port ${PORT}`);
});
