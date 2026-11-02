@AGENTS.md

Claude Code specifics:
- Headless: `envs/base/bin/any2sim run projects/<id> --agent claude [--model M --effort E]`. Interactive: start
  `claude` in this directory; `.mcp.json` provides the `any2sim` MCP server.
- Skills: `skills/<stage>/SKILL.md`, linked into `.claude/skills/`. Evaluator: `agents/stage-evaluator.md`, linked
  into `.claude/agents/`; delegate evaluations with the Agent tool, `subagent_type: stage-evaluator`.
- View evidence images with the Read tool.
