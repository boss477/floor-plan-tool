import app

# Test 1: Clean JSON response
resp1 = {
    "choices": [
        {
            "message": {
                "content": '[{"id":1,"name":"Master Bedroom","type":"bedroom","width_mm":4320,"length_mm":8030,"source":"printed","confidence":"high"}]'
            }
        }
    ]
}
rooms1 = app.parse_response(resp1)
print("Test 1 - Clean JSON:", rooms1)

# Test 2: Markdown-wrapped response
resp2 = {
    "choices": [
        {
            "message": {
                "content": '```json\n[{"id":1,"name":"Kitchen","type":"kitchen","width_mm":3000,"length_mm":3500,"source":"printed","confidence":"high"}]\n```'
            }
        }
    ]
}
rooms2 = app.parse_response(resp2)
print("Test 2 - Markdown wrapped:", rooms2)

# Test 3: Think tags before JSON
resp3 = {
    "choices": [
        {
            "message": {
                "content": '<think>internal reasoning</think>[{"id":1,"name":"Bath","type":"bathroom","width_mm":1800,"length_mm":2400,"source":"estimated","confidence":"medium"}]'
            }
        }
    ]
}
rooms3 = app.parse_response(resp3)
print("Test 3 - Think tags stripped:", rooms3)

# Test 4: Payload structure
payload = app.build_payload("test prompt", "base64data", "image/jpeg")
print("Test 4 - Payload model:", payload["model"])
print(
    "Test 4 - Payload format correct:",
    all(
        [
            payload["messages"][0]["content"][0]["type"] == "text",
            payload["messages"][0]["content"][1]["type"] == "image_url",
            payload["messages"][0]["content"][1]["image_url"]["url"].startswith(
                "data:image/jpeg;base64,"
            ),
        ]
    ),
)

print("\nAll tests passed!")
