from werkzeug.security import generate_password_hash

# The word you want to hash
my_word = "osas"

# Generate the password hash
hashed_word = generate_password_hash(my_word)

print(f"Original word: {my_word}")
print(f"Hashed word: {hashed_word}")
