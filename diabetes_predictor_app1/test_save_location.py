import os

# This script will solve the mystery of the save location.

# 1. Print the "Current Working Directory" (CWD)
# This is the location your terminal is currently running from.
cwd_path = os.getcwd()
print(f"👉 Right now, your terminal is running from this directory:")
print(f"   {cwd_path}\n")

# 2. Define the 'models' directory path relative to the CWD
models_dir = os.path.join(cwd_path, "models")
print(f"👉 Therefore, the script is trying to create the 'models' folder here:")
print(f"   {models_dir}\n")

# 3. Create the directory
try:
    os.makedirs(models_dir, exist_ok=True)
    print("✅ 'models' directory has been created (or already exists).\n")
except Exception as e:
    print(f"🔴 ERROR creating directory: {e}")
    exit()

# 4. Create a simple test file inside that directory
test_file_path = os.path.join(models_dir, "hello_from_python.txt")
try:
    with open(test_file_path, "w") as f:
        f.write("If you can see this file, this is the correct 'models' folder!")
    print(f"✅ A test file was successfully saved to:")
    print(f"   {test_file_path}\n")
    print("--- 💡 NEXT STEP 💡 ---")
    print("Copy the full path above and paste it into your computer's File Explorer address bar to see the file.")

except Exception as e:
    print(f"🔴 ERROR saving the test file: {e}")