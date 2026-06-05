import re

def test_regex():
    cases = [
        "0912 345 678",
        "0912-345-678",
        "+84912345678",
        "84912345678",
        "(0912) 345.678",
        "091234567",
        "09123456789",
        "02438222222",
        "0123456789",
        "nha so 0987654321",
        "ma don 0912345678",
        "Lan 0901234567, lien he shop 0987654321",
        "0901234567 0987654321",  # Should be TWO candidates
        "(+84) 912 345 678",
    ]
    
    # regex to capture wrapper span
    # prefix: +84, 84, or 0
    # followed by 6 to 11 instances of (optional non-digits followed by a digit)
    pattern = re.compile(r'(?<!\d)\(?(?:\+?84|0)(?:[\-\.\(\)]*\s*[\-\.\(\)]*\d){7,11}(?!\d)\)?')
    for case in cases:
        matches = [m.group(0) for m in pattern.finditer(case)]
        print(f"{case} -> {matches}")

test_regex()
