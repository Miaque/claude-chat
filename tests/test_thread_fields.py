from models.thread import Thread, Threads
import uuid

def test_get_specific_fields():
    # 测试 UUID（你可以替换为实际存在的 thread_id）
    test_thread_id = "1a90e93f-f802-4519-8645-c1f983c84e77"
    
    print("=== 测试查询指定字段 ===")
    
    # 1. 查询单个字段
    print("\n1. 查询单个字段 project_id:")
    project_id = Threads.get_by_id(test_thread_id, Thread.project_id)
    print(f"project_id: {project_id}")
    
    # 2. 查询多个字段
    print("\n2. 查询多个字段 (account_id, created_at, meta):")
    thread_data = Threads.get_by_id(
        test_thread_id, 
        Thread.account_id, 
        Thread.created_at, 
        Thread.meta
    )
    print(f"thread_data: {thread_data}")
    
    # 3. 查询完整对象（对比）
    print("\n3. 查询完整对象:")
    full_thread = Threads.get_by_id(test_thread_id)
    print(f"full_thread: {full_thread}")
    
    # 4. 查询不存在的字段
    print("\n4. 查询不存在的 ID:")
    not_found = Threads.get_by_id(str(uuid.uuid4()), Thread.project_id)
    print(f"not_found: {not_found}")

if __name__ == "__main__":
    test_get_specific_fields()
