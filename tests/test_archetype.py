import sys, pathlib, asyncio
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG


def test_choose_class_retries_until_two_words(monkeypatch):
    seq = iter(["Wise Owl Wizard", "Arcane Scholar of Dawn", "Arcane Scholar"])

    async def fake_chat(messages, options=None):
        return next(seq)

    monkeypatch.setattr(oRPG, "ollama_chat", fake_chat)
    result = asyncio.run(oRPG.choose_class_with_ollama("A scholarly mage seeking ancient tomes."))
    assert result == "Arcane Scholar"
    assert len(result.split()) <= 2


def test_choose_class_strips_prefix_and_quotes(monkeypatch):
    async def fake_chat(messages, options=None):
        return 'Class:  "Night   Stalker"\n'

    monkeypatch.setattr(oRPG, "ollama_chat", fake_chat)
    result = asyncio.run(oRPG.choose_class_with_ollama("shadowy background"))
    assert result == "Night Stalker"


def test_choose_class_preserves_hyphen_and_apostrophe(monkeypatch):
    async def fake_chat(messages, options=None):
        return "`King's-Guard`"

    monkeypatch.setattr(oRPG, "ollama_chat", fake_chat)
    result = asyncio.run(oRPG.choose_class_with_ollama("royal guard"))
    assert result == "King's-Guard"


def test_choose_class_fallback_to_first_two_words(monkeypatch):
    async def fake_chat(messages, options=None):
        return "Too Many Words Here"

    monkeypatch.setattr(oRPG, "ollama_chat", fake_chat)
    result = asyncio.run(oRPG.choose_class_with_ollama("bg"))
    assert result == "Too Many"


def test_choose_class_fallback_to_adventurer_when_empty(monkeypatch):
    async def fake_chat(messages, options=None):
        return ""

    monkeypatch.setattr(oRPG, "ollama_chat", fake_chat)
    result = asyncio.run(oRPG.choose_class_with_ollama(""))
    assert result == "Adventurer"


def test_choose_class_truncates_to_40_chars(monkeypatch):
    long_two_words = ("A" * 50) + " " + ("B" * 50)

    async def fake_chat(messages, options=None):
        return long_two_words

    monkeypatch.setattr(oRPG, "ollama_chat", fake_chat)
    result = asyncio.run(oRPG.choose_class_with_ollama("bg"))
    assert result == "A" * 40


def test_choose_class_strips_various_labels(monkeypatch):
    seq = iter(["role: Shadow Blade", "Archetype: Mystic Warrior"])  # first is already <=2 words

    async def fake_chat(messages, options=None):
        return next(seq)

    monkeypatch.setattr(oRPG, "ollama_chat", fake_chat)
    r1 = asyncio.run(oRPG.choose_class_with_ollama("bg"))
    assert r1 == "Shadow Blade"
    r2 = asyncio.run(oRPG.choose_class_with_ollama("bg"))
    assert r2 == "Mystic Warrior"
