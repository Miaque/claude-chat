def test_uuid():
    import uuid

    print(uuid.UUID(int=0))
    print(uuid.uuid4())
    print(str(uuid.uuid4())[:8])
