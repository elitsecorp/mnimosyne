"""Prompt templates for chat and memory extraction."""

from __future__ import annotations

CHAT_SYSTEM_PROMPT = """You are a helpful assistant with access to long-term memory.

Your memory system stores two kinds of information:
1. Relevant past conversations (vector memories)
2. Structured knowledge about entities and relationships (ontology graph)

Use this information to provide accurate, context-aware responses.
If the memory context contains relevant information, incorporate it naturally.
If no relevant memories are found, respond based on your general knowledge.
Do not mention the memory system to the user.

Formatting rules:
- Use markdown formatting in your responses
- Use **bold** for emphasis on key terms
- Use bullet points (- or *) for lists of items
- Use headers (## or ###) for sections when organizing information
- Use `code` for entity names, technical terms, or specific values
- Use > blockquotes for notable quotes or important facts
- Keep paragraphs short and well-spaced
- When describing entities, organize information clearly with labels and values
- When listing relationships, use a structured format
- End with a clear summary when appropriate"""

EXTRACTION_SYSTEM_PROMPT = """You are a knowledge extraction engine. Given a conversation between a user and an assistant, extract structured knowledge.

Output ONLY valid JSON matching this exact schema:
{
  "entities": [
    {"name": "entity name", "type": "entity type", "confidence": 0.95}
  ],
  "relationships": [
    {"subject": "entity A", "predicate": "relationship_type", "object": "entity B", "confidence": 0.94}
  ],
  "facts": [
    {"subject": "subject", "predicate": "predicate", "object": "object"}
  ]
}

Rules:
- Entity types should be lowercase: person, animal, place, organization, object, concept, event, etc.
- Predicates must use snake_case: works_for, likes, owns, located_in, etc.
- Extract only clear, explicit facts. Do not infer.
- confidence values between 0.0 and 1.0
- No explanations. Only JSON. No markdown fencing."""


def build_chat_messages(
    conversation: list[dict[str, str]],
    vector_memories: str,
    ontology_facts: str,
    user_message: str,
) -> list[dict[str, str]]:
    """Build the message list for chat completion.

    Args:
        conversation: Previous messages in the current session.
        vector_memories: Formatted vector search results.
        ontology_facts: Formatted graph/ontology context.
        user_message: The current user message.

    Returns:
        List of message dicts for the LLM API.
    """
    context_parts = []
    if vector_memories:
        context_parts.append(f"[Relevant Memories]\n{vector_memories}")
    if ontology_facts:
        context_parts.append(f"[Known Facts]\n{ontology_facts}")
    context_block = "\n\n".join(context_parts) if context_parts else "No relevant memories found."

    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]

    if conversation:
        messages.extend(conversation[-20:])

    messages.append({
        "role": "user",
        "content": f"Memory Context:\n{context_block}\n\nUser: {user_message}",
    })

    return messages


def build_extraction_messages(user_message: str, assistant_response: str) -> list[dict[str, str]]:
    """Build the message list for memory extraction.

    Args:
        user_message: What the user said.
        assistant_response: What the assistant replied.

    Returns:
        List of message dicts for the LLM API.
    """
    conversation = (
        f"User: {user_message}\nAssistant: {assistant_response}"
    )
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": conversation},
    ]
