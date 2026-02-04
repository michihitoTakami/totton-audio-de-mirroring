# AI Integration Documentation

## Overview

This document describes how various AI tools and agents are integrated into the Neural-DeRinger development workflow.

---

## Supported AI Tools

### 1. Claude Code

**Primary AI development assistant**

- **Configuration**: [CLAUDE.md](../CLAUDE.md)
- **Usage**: Command-line interface for AI-assisted development
- **Language**: Think in English, respond in Japanese

**Key Features**:
- Code generation with type hints and docstrings
- Test generation
- Refactoring assistance
- Git workflow automation

### 2. GitHub Copilot

**Code completion and suggestion**

- **Configuration**: [.github/copilot-instructions.md](../.github/copilot-instructions.md)
- **Usage**: Inline code suggestions in IDE
- **Integration**: Automatic via GitHub account

**Key Features**:
- Function completion
- Docstring generation
- Test case suggestions
- Import statement completion

### 3. Cursor

**AI-powered code editor**

- **Configuration**: Uses `.github/copilot-instructions.md`
- **Usage**: Native AI chat and editing features
- **Integration**: Reads project context automatically

**Key Features**:
- Multi-file editing
- Codebase-aware chat
- Inline AI editing
- Documentation generation

### 4. Generic AI Agents

**Any AI assistant (ChatGPT, Claude Web, etc.)**

- **Configuration**: [AGENTS.md](../AGENTS.md)
- **Usage**: Copy/paste code or provide context
- **Integration**: Manual context sharing

**Key Features**:
- Code review
- Architecture discussion
- Debugging assistance
- Learning and explanation

---

## Documentation Hierarchy

```
/home/michihito/Working/totton-audio-up-sample-test/
├── README.md                          # Project overview
├── CLAUDE.md                          # Main development guide (PRIMARY)
├── AGENTS.md                          # AI agents quick reference
├── .github/
│   └── copilot-instructions.md       # GitHub Copilot configuration
├── .claude/
│   └── rules/
│       ├── testing.md                # Testing guidelines (DETAILED)
│       ├── coding-style.md           # Code style rules (DETAILED)
│       └── security.md               # Security best practices (DETAILED)
└── docs/
    └── AI_INTEGRATION.md             # This file (OVERVIEW)
```

### Reading Order for New AI Agents

1. **[AGENTS.md](../AGENTS.md)** - Quick start and core principles
2. **[CLAUDE.md](../CLAUDE.md)** - Complete development guide
3. **[.claude/rules/](../.claude/rules/)** - Specific domain guidelines
4. **[.github/copilot-instructions.md](../.github/copilot-instructions.md)** - Code generation templates

---

## Integration by Use Case

### Use Case 1: Writing New Feature

**Best Tool**: Claude Code or Cursor

**Workflow**:
1. Read [CLAUDE.md](../CLAUDE.md) for context
2. Follow development workflow (create branch, etc.)
3. Generate code with type hints and docstrings
4. Generate corresponding tests
5. Run quality checks before commit

**Example Command** (Claude Code):
```
I need to implement a Bessel filter function in src/neural_deringer/data/filters.py
```

### Use Case 2: Code Completion

**Best Tool**: GitHub Copilot

**Workflow**:
1. Start typing function signature
2. Accept Copilot suggestions
3. Review and modify generated code
4. Ensure type hints and docstrings are complete

**Example**:
```python
# Type this:
def bessel_filter(signal: np.ndarray, cutoff_freq: float

# Copilot suggests:
def bessel_filter(signal: np.ndarray, cutoff_freq: float, order: int = 8) -> np.ndarray:
    """Apply Bessel filter for zero-ringing upsampling.

    Args:
        signal: Input audio signal
        cutoff_freq: Cutoff frequency in Hz
        order: Filter order (default: 8)

    Returns:
        Filtered signal with minimal phase distortion
    """
```

### Use Case 3: Test Generation

**Best Tool**: Claude Code or GitHub Copilot

**Workflow**:
1. Implement function
2. Ask AI to generate tests
3. Review test coverage
4. Add edge case tests

**Example Prompt** (Claude Code):
```
Generate comprehensive tests for the bessel_filter function, including:
- Basic functionality
- Edge cases (empty input, single sample)
- Invalid inputs (negative cutoff, etc.)
- Parametrized tests for various orders
```

### Use Case 4: Code Review

**Best Tool**: Generic AI Agent (ChatGPT, Claude Web)

**Workflow**:
1. Copy code to AI chat
2. Provide [AGENTS.md](../AGENTS.md) context
3. Ask specific questions
4. Apply suggested improvements

**Example Prompt**:
```
Review this code for the Neural-DeRinger project.
Context: [paste AGENTS.md content]

Code:
[paste your code]

Check for:
- Type hints and docstrings
- Input validation
- Immutability
- Error handling
```

### Use Case 5: Architecture Discussion

**Best Tool**: Generic AI Agent or Claude Code

**Workflow**:
1. Describe problem and constraints
2. Provide project context
3. Discuss trade-offs
4. Validate against physics principles

**Example Prompt**:
```
I need to design a U-Net architecture for audio upsampling in the Neural-DeRinger project.

Requirements:
- Input: 1D audio signal (time series)
- Output: Same length signal with reduced ringing
- Preserve frequencies up to Nyquist limit
- Real-time capable

What architecture would you recommend?
```

---

## Configuration Management

### Environment Setup for AI Tools

#### 1. Claude Code

Install and configure:
```bash
# Install (if not already installed)
npm install -g @anthropic/claude-code

# Authenticate
claude auth login

# Initialize project
cd /path/to/totton-audio-up-sample-test
claude init
```

#### 2. GitHub Copilot

Enable in IDE:
- **VS Code**: Install "GitHub Copilot" extension
- **Cursor**: Built-in, enable in settings
- **JetBrains**: Install "GitHub Copilot" plugin

Configuration:
- Uses `.github/copilot-instructions.md` automatically
- No additional setup needed

#### 3. Cursor

Install and configure:
```bash
# Download from https://cursor.sh/
# Open project directory
cursor /path/to/totton-audio-up-sample-test
```

Settings:
- Enable AI features in preferences
- Set up rules to reference `.github/copilot-instructions.md`

---

## Best Practices

### For AI-Generated Code

1. **Always Review**: Never blindly accept AI suggestions
2. **Verify Physics**: Ensure physical assumptions are correct
3. **Run Tests**: Test AI-generated code thoroughly
4. **Check Security**: Validate no security vulnerabilities
5. **Maintain Style**: Ensure consistency with project guidelines

### For AI Conversations

1. **Provide Context**: Share [AGENTS.md](../AGENTS.md) or relevant docs
2. **Be Specific**: Ask concrete questions with clear constraints
3. **Iterate**: Refine prompts based on responses
4. **Validate**: Cross-check AI responses with documentation

### For Documentation

1. **Keep Updated**: Update AI instructions when patterns change
2. **Be Explicit**: Don't assume AI knows project specifics
3. **Reference Physics**: Always include physical basis for decisions
4. **Link Docs**: Cross-reference related documentation

---

## Common Pitfalls

### ❌ Pitfall 1: Skipping Type Hints

**Problem**:
```python
# AI generates:
def process(signal, sr):
    return filtered
```

**Solution**: Always add type hints
```python
def process(signal: np.ndarray, sr: int) -> np.ndarray:
    return filtered
```

### ❌ Pitfall 2: Missing Validation

**Problem**:
```python
def upsample(signal, ratio):
    return signal.repeat(ratio)
```

**Solution**: Add input validation
```python
def upsample(signal: np.ndarray, ratio: int) -> np.ndarray:
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    return signal.repeat(ratio)
```

### ❌ Pitfall 3: Mutable Operations

**Problem**:
```python
def add_noise(signal, noise_level):
    signal += np.random.randn(*signal.shape) * noise_level
    return signal
```

**Solution**: Use immutable pattern
```python
def add_noise(signal: np.ndarray, noise_level: float) -> np.ndarray:
    noise = np.random.randn(*signal.shape) * noise_level
    return signal + noise
```

### ❌ Pitfall 4: Missing Physical Basis

**Problem**:
```python
def apply_filter(signal, cutoff):
    """Apply filter to signal."""
    return filtered
```

**Solution**: Add Physical Basis section
```python
def apply_filter(signal: np.ndarray, cutoff: float) -> np.ndarray:
    """Apply Bessel filter to signal.

    Physical Basis:
        Bessel filters have maximally flat group delay, resulting in
        zero overshoot in step response and minimal ringing artifacts.
    """
    return filtered
```

---

## Troubleshooting

### Issue 1: AI Not Following Project Conventions

**Symptoms**: Generated code lacks type hints, docstrings, or validation

**Solution**:
1. Explicitly reference [AGENTS.md](../AGENTS.md) in prompt
2. Ask AI to regenerate with specific requirements
3. Update AI instructions if pattern is unclear

### Issue 2: AI Suggests Insecure Code

**Symptoms**: Hardcoded secrets, `eval()` usage, missing validation

**Solution**:
1. Reference [.claude/rules/security.md](../.claude/rules/security.md)
2. Explicitly ask for security review
3. Run security linters (bandit, safety)

### Issue 3: AI Generates Overly Complex Code

**Symptoms**: Functions >50 lines, deep nesting, premature optimization

**Solution**:
1. Reference [.claude/rules/coding-style.md](../.claude/rules/coding-style.md)
2. Ask AI to simplify and split into smaller functions
3. Apply YAGNI principle (You Aren't Gonna Need It)

### Issue 4: AI Doesn't Understand Physics Context

**Symptoms**: Suggests frequency generation above Nyquist, ignores phase

**Solution**:
1. Provide explicit physics constraints in prompt
2. Reference [CLAUDE.md Physical Basis sections](../CLAUDE.md)
3. Validate results with theoretical predictions

---

## Extending AI Integration

### Adding New AI Tool Support

1. **Create configuration file** in appropriate location:
   - IDE-specific: `.vscode/`, `.idea/`, etc.
   - Generic: `.github/` or root directory

2. **Reference existing docs** to maintain consistency:
   ```markdown
   # New Tool Instructions

   ## Overview
   [Tool description]

   ## Project Context
   See [AGENTS.md](../AGENTS.md) for full context.

   ## Tool-Specific Guidelines
   [Tool-specific rules]
   ```

3. **Update this document** with new tool information

4. **Test integration** with sample prompts and workflows

### Adding New Guidelines

1. **Determine scope**:
   - General: Update [CLAUDE.md](../CLAUDE.md)
   - Specific domain: Create/update `.claude/rules/<domain>.md`
   - Quick reference: Update [AGENTS.md](../AGENTS.md)

2. **Ensure discoverability**:
   - Link from [README.md](../README.md)
   - Cross-reference in related docs
   - Add to this file's hierarchy section

3. **Validate with AI tools**:
   - Test that AI assistants can follow new guidelines
   - Iterate based on observed behavior

---

## Metrics and Monitoring

### Code Quality Metrics

Track AI-generated code quality:
- **Test Coverage**: Aim for 80%+
- **Type Hint Coverage**: Aim for 100%
- **Docstring Coverage**: Aim for 100%
- **Linting Errors**: Aim for 0

### AI Usage Metrics

Monitor AI effectiveness:
- **Acceptance Rate**: % of AI suggestions accepted
- **Edit Rate**: % of AI code requiring modification
- **Bug Rate**: Bugs in AI-generated code vs human-written
- **Time Savings**: Development time with vs without AI

---

## Future Improvements

### Planned Features

1. **Automated Context Injection**: Auto-load relevant docs for AI tools
2. **Custom AI Actions**: Pre-configured workflows for common tasks
3. **Quality Gates**: Automated checks for AI-generated code
4. **Physics Validation**: Automated verification of physical assumptions

### Research Areas

1. **Physics-Aware AI**: Train AI on signal processing theory
2. **Test Co-Generation**: Simultaneous code and test generation
3. **Architecture Search**: AI-assisted neural architecture design
4. **Performance Optimization**: AI-guided performance tuning

---

## Additional Resources

- [Claude Code Documentation](https://docs.anthropic.com/claude/docs/claude-code)
- [GitHub Copilot Documentation](https://docs.github.com/en/copilot)
- [Cursor Documentation](https://cursor.sh/docs)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

## Questions?

For AI integration issues or suggestions:
1. Check [AGENTS.md](../AGENTS.md) for quick reference
2. Review [CLAUDE.md](../CLAUDE.md) for full guidelines
3. Open an issue with `[AI]` prefix
4. Ask AI assistant with proper context

---

**Last Updated**: 2026-01-27
