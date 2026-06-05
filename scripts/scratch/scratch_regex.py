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
        "ma don 0912345678"
    ]
    
    # regex to capture wrapper span
    pattern = re.compile(r'(?<!\d)\(?(?:\+?84|0)[\d\s\-\.\(\)]*\d(?!\d)\)?')
    for case in cases:
        matches = [m.group(0) for m in pattern.finditer(case)]
        print(f"{case} -> {matches}")

test_regex()
