from app.classifier.preprocessing import clean_message, mask_pii, remove_quoted_and_signature
from app.classifier.service import TicketClassifier
from app.llm import MockLLM


def test_pii_is_masked_before_anything_else_sees_it():
    """Emails, phone numbers and IBANs are replaced by placeholders."""
    text = mask_pii("Mail me at a.b@example.com, call +91 98765 43210, IBAN DE89 3704 0044 0532 0130 00")
    assert "example.com" not in text and "98765" not in text and "DE89" not in text
    assert "<EMAIL>" in text and "<NUMBER>" in text and "<IBAN>" in text


def test_quoted_reply_and_signature_are_removed():
    """Signatures and quoted replies are cut; only the newest message remains."""
    body = "The compressor shows E104.\nRegards,\nRahul\n> On Monday you wrote: old text"
    assert remove_quoted_and_signature(body) == "The compressor shows E104."


def test_clean_message_strips_html_and_lowercases():
    """HTML is stripped and text lower-cased."""
    assert clean_message("Invoice", "<p>Charged <b>twice</b></p>") == "invoice charged twice"


def test_confident_messages_use_the_fast_model():
    """A clear billing message is decided by the fast model without the LLM."""
    result = TicketClassifier(MockLLM()).classify("Refund", "Please refund the duplicate payment on invoice 4471")
    assert result.category == "billing" and result.source == "fast_model"


class FailingLLM:
    """Fake LLM simulating a provider outage."""
    def complete(self, system, user, *, json_mode=False):
        """Always raise, as a timed-out provider would."""
        raise TimeoutError("provider down")


class GarbageLLM:
    """Fake LLM returning JSON that violates the schema (unknown category, confidence > 1)."""
    def complete(self, system, user, *, json_mode=False):
        """Return the invalid JSON."""
        return '{"category": "urgent", "confidence": 2}'  # not in the enum, out of range


def test_unclear_message_goes_to_human_review_when_llm_is_down():
    """If the LLM is down, an unclear message goes to human review instead of crashing."""
    result = TicketClassifier(FailingLLM()).classify("", "hello")
    assert result.category == "needs_review" and result.source == "human_review"


def test_invalid_llm_output_is_rejected_not_trusted():
    """Invalid LLM output is rejected by the LLMLabel schema, never trusted."""
    result = TicketClassifier(GarbageLLM()).classify("", "hello")
    assert result.category == "needs_review"
