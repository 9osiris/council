"""render a run result as readable markdown."""


def render_transcript(transcript):
    """transcript is a list of (speaker, text) pairs."""
    lines = []
    for speaker, text in transcript:
        lines.append("**%s:** %s" % (speaker, text.strip()))
        lines.append("")
    return "\n".join(lines).rstrip()


def to_markdown(result):
    usage = result.usage_totals()
    lines = [
        "# council run: %s" % result.pattern,
        "",
        "task: %s" % result.task,
        "",
        "## transcript",
        "",
    ]
    for t in result.turns:
        tag = " (%s)" % t.kind if t.kind != "say" else ""
        lines.append("### %s%s" % (t.speaker, tag))
        lines.append("")
        lines.append(t.text.strip())
        lines.append("")
    lines.append("## final answer")
    lines.append("")
    lines.append(result.answer.strip())
    lines.append("")
    lines.append("## stats")
    lines.append("")
    lines.append("- turns: %d" % len(result.turns))
    lines.append("- seconds: %.1f" % result.seconds)
    lines.append("- tokens: %d in / %d out" % (
        usage["prompt_tokens"], usage["completion_tokens"]))
    spend = result.spend_by_agent()
    if spend:
        lines.append("- spend by agent: %s" % ", ".join(
            "%s=%d" % (name, row["total_tokens"])
            for name, row in spend.items()))
    if result.extra:
        lines.append("")
        lines.append("## extra")
        lines.append("")
        for k, v in result.extra.items():
            lines.append("- %s: %s" % (k, v))
    return "\n".join(lines).rstrip() + "\n"
