"""
Diagnostic script to check Celery task registration
"""
import sys

print("=" * 60)
print("CELERY TASK REGISTRATION DIAGNOSTIC")
print("=" * 60)

# Test 1: Import celery_app
print("\n1. Importing celery_app...")
try:
    from celery_app import celery_app
    print("   ✓ celery_app imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import celery_app: {e}")
    sys.exit(1)

# Test 2: Import task
print("\n2. Importing generate_3d_asset task...")
try:
    from tasks import generate_3d_asset
    print("   ✓ generate_3d_asset imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import generate_3d_asset: {e}")
    sys.exit(1)

# Test 3: Check if task is registered
print("\n3. Checking task registration...")
print(f"\n   Registered tasks in celery_app:")
for task_name in sorted(celery_app.tasks.keys()):
    print(f"   - {task_name}")

print("\n4. Looking for generate_3d_asset task...")
task_found = False
for task_name in celery_app.tasks.keys():
    if 'generate_3d_asset' in task_name:
        print(f"   ✓ Found: {task_name}")
        task_found = True
        
        # Get task object
        task_obj = celery_app.tasks[task_name]
        print(f"   Task object: {task_obj}")
        
        # Check signature
        import inspect
        sig = inspect.signature(task_obj.run)
        print(f"   Signature: {sig}")

if not task_found:
    print("   ✗ generate_3d_asset task NOT FOUND!")
    print("\n   This is the problem! The task needs to be registered.")
    print("   Check that celery_app.py imports tasks.py")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
