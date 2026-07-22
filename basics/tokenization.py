import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")

print("Vocab Size of OpenAI tiktoken", encoder.n_vocab) # 2,00,019


text = "My name is Vivek Patel."  #[5444, 1308, 382, 118495, 74, 122760, 13]    

tokens = encoder.encode(text)
print(tokens) #[5444, 1308, 382, 118495, 74, 122760, 13]   

print(encoder.decode([5444, 1308, 382, 118495, 74, 122760, 13])) #My name is Vivek Patel.