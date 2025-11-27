from models.thread import Thread, Threads


def test_query_thread():
    thread = Threads.get_by_id("1a90e93f-f802-4519-8645-c1f983c84e77")
    print(thread)
    print("\n")

    thread = Threads.get_by_id("a25090dd-90ad-439f-977e-16a499335fd4")
    print(thread)
    if thread:
        print(f"{thread.project_id!r}")
        print(f"{thread.account_id!r}")
        print(f"{thread.meta!r}")
        print("\n")

    thread = Threads.get_by_id(
        "a25090dd-90ad-439f-977e-16a499335fd4",
        Thread.project_id,
        Thread.account_id,
        Thread.meta,
    )
    if thread:
        print(f"{thread.project_id!r}")
        print(f"{thread.account_id!r}")
        print(f"{thread.meta!r}")
        print("\n")


def test_query_thread_id():
    response = Threads.get_by_id(
        "a25090dd-90ad-439f-977e-16a499335fd4", Thread.account_id
    )

    if response:
        print(response.account_id)
