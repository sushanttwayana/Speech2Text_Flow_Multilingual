import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict

class MobileTranslator:
    def __init__(self, model_size="distilled-600M"):
        self.model_name = f"facebook/nllb-200-{model_size}"
        self.language_codes = {
            "english": "eng_Latn",
            "hindi": "hin_Deva",
            "malay": "zsm_Latn"
        }
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self._cache_language_tokens()

    def _cache_language_tokens(self):
        self.lang_token_ids = {}
        for lang_name, lang_code in self.language_codes.items():
            token_id = self.tokenizer.convert_tokens_to_ids([lang_code])[0]
            self.lang_token_ids[lang_name] = token_id

    def translate(self, text: str, source_lang: str, target_lang: str) -> Dict:
        if source_lang.lower() not in self.language_codes or target_lang.lower() not in self.language_codes:
            return {"error": f"Unsupported language(s): {source_lang}, {target_lang}", "success": False}
        source_code = self.language_codes[source_lang.lower()]
        target_code = self.language_codes[target_lang.lower()]
        input_text = f"{source_code} {text}"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        target_token_id = self.lang_token_ids[target_lang.lower()]
        generation_args = {
            "forced_bos_token_id": target_token_id,
            "max_length": 512,
            "num_beams": 4,
            "early_stopping": True,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id
        }
        with torch.no_grad():
            generated_tokens = self.model.generate(**inputs, **generation_args)
        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        # Remove target language code from output if present
        for lang_code in self.language_codes.values():
            translated_text = translated_text.replace(lang_code, "").strip()
        return {
            "original_text": text,
            "translated_text": translated_text,
            "source_language": source_lang,
            "target_language": target_lang,
            "success": True
        }
