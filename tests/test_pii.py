import pytest

from gateway.guardrails.pii import Vault, find_pii, luhn_valid, mask_pii, thai_id_valid


def make_thai_id(first12: str) -> str:
    total = sum(int(d) * (13 - i) for i, d in enumerate(first12))
    return first12 + str((11 - total % 11) % 10)


VALID_ID = make_thai_id("110170023070")


def kinds(text):
    return [(f.kind, f.value) for f in find_pii(text)]


def test_thai_id_checksum():
    assert thai_id_valid(VALID_ID)
    wrong = VALID_ID[:-1] + str((int(VALID_ID[-1]) + 1) % 10)
    assert not thai_id_valid(wrong)


@pytest.mark.parametrize("fmt", ["{}", "{0}-{1}-{2}-{3}-{4}", "{0} {1} {2} {3} {4}"])
def test_thai_id_formats(fmt):
    d = VALID_ID
    parts = (d[0], d[1:5], d[5:10], d[10:12], d[12])
    text = "เลขบัตรประชาชน " + (fmt.format(*parts) if fmt != "{}" else d) + " ครับ"
    assert [k for k, _ in kinds(text)] == ["THAI_ID"]


def test_random_13_digits_are_not_an_id():
    bad = VALID_ID[:-1] + str((int(VALID_ID[-1]) + 1) % 10)
    assert kinds(f"order {bad} shipped") == []


@pytest.mark.parametrize("phone", ["081-234-5678", "0812345678", "+66 81 234 5678", "02-123-4567", "053-123-456"])
def test_phones(phone):
    assert kinds(f"โทร {phone} ได้เลย") == [("PHONE", phone)]


def test_email_card_bank():
    assert kinds("mail me at a.b+c@example.co.th") == [("EMAIL", "a.b+c@example.co.th")]
    assert kinds("card 4111 1111 1111 1111") == [("CARD", "4111 1111 1111 1111")]
    assert kinds("card 4111 1111 1111 1112") == []  # fails Luhn
    assert kinds("โอนเข้าบัญชี 123-4-56789-0 นะ") == [("BANK_ACCOUNT", "123-4-56789-0")]
    assert kinds("เลขบัญชี 1234567890") == [("BANK_ACCOUNT", "1234567890")]


def test_luhn():
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("ผมชื่อนายสมชาย ใจดี โทรมาได้", "สมชาย ใจดี"),
        ("นางสาววิภาวดีรังสิตมาแล้ว", "วิภาวดีรังสิต"),
        ("ด.ช.ต้น สุขใจ", "ต้น สุขใจ"),
        ("Please call Dr. Jane Smith tomorrow", "Jane Smith"),
    ],
)
def test_names_after_titles(text, name):
    assert ("PERSON", name) in kinds(text)


@pytest.mark.parametrize(
    "text", ["นายกรัฐมนตรีแถลงข่าว", "นางพยาบาลใจดี", "นายจ้างจ่ายเงินเดือน", "นางฟ้าตัวน้อย", "ไปหานายอำเภอ", "คุณช่วยสรุปให้หน่อย"]
)
def test_titles_inside_words_are_not_names(text):
    assert kinds(text) == []


def test_vault_redacts_and_restores():
    vault = Vault()
    r = vault.redact("นายสมชาย ใจดี เบอร์ 081-234-5678 และ 081-234-5678 อีกครั้ง")
    assert r.text == "นาย[PERSON_1] เบอร์ [PHONE_1] และ [PHONE_1] อีกครั้ง"
    assert vault.total == 2
    assert vault.restore("โทรหา [PERSON_1] ที่ [PHONE_1] [PHONE_9]") == "โทรหา สมชาย ใจดี ที่ 081-234-5678 [PHONE_9]"


def test_mask_is_not_restorable():
    text, n = mask_pii("ติดต่อ 02-123-4567")
    assert (text, n) == ("ติดต่อ [REDACTED_PHONE]", 1)
    assert Vault().restore(text) == text


def test_surname_with_stop_word_inside():
    assert ("PERSON", "ธนพล วงศ์ใหญ่") in kinds("ลูกค้าชื่อนายธนพล วงศ์ใหญ่ โทร 081-111-2222")
