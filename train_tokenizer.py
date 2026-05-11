from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from tokenizers.normalizers import NFKC
from transformers import PreTrainedTokenizerFast

# 1️⃣ Initialize GPT-2–style Byte-Level BPE
tokenizer = Tokenizer(models.BPE(unk_token="<|endoftext|>"))
tokenizer.normalizer = NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 2️⃣ Train with GPT-2-like special tokens
trainer = trainers.BpeTrainer(
    vocab_size=10_000,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<|endoftext|>"],  # GPT-2 uses this for BOS/EOS/UNK
    continuing_subword_prefix="Ġ",     # space marker used by GPT-2
)

# 3️⃣ Train on your TinyStories text corpus
files = ["/shared_data0/hnwong/cache/TinyStories-train.txt"]  # single file with all your text
tokenizer.train(files=files, trainer=trainer)

# 4️⃣ Enable byte-level post-processing (for proper GPT-2 detokenization)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

# 5️⃣ Save the raw tokenizer
tokenizer.save("/shared_data0/hnwong/cache/tokenizer_tinystories_gpt2_10k.json")

# 6️⃣ Wrap it for Hugging Face compatibility
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="/shared_data0/hnwong/cache/tokenizer_tinystories_gpt2_10k.json",
    tokenizer_type="GPT2Tokenizer",
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
    unk_token="<|endoftext|>",
    model_max_length=2048,
    add_prefix_space=False,
)

hf_tokenizer.save_pretrained("/shared_data0/hnwong/cache/tokenizer_tinystories_gpt2_10k")

print("✅ Trained GPT-2-style tokenizer saved at tokenizer_tinystories_gpt2_10k/")
print("Vocab size:", len(hf_tokenizer))
