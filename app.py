import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel
import torch
from search_func import query_cosine_similarity_chunks, query_cosine_similarity_klasif1, \
    query_cosine_similarity_klasif2, search_no_polinom
from itertools import product
import psycopg2

# Загрузка токенизатора и модели для векторизации
tokenizer_vectors = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
model_vectors = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")

torch.manual_seed(42)

# Название модели
model_name = "AnatoliiPotapov/T-lite-instruct-0.1"

# Настройка конфигурации для квантования
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16  # Установка типа данных для ускорения
)

# Загрузка токенизатора и модели с квантованием в 4 бита
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,  # Передача конфигурации квантования
    torch_dtype=torch.float16
)
model.eval()

app = FastAPI()


class Request(BaseModel):
    question: str


class Response(BaseModel):
    answer: str
    class_1: str
    class_2: str


@app.get("/")
def index():
    return {"message": "Интеллектуальный помощник оператора службы поддержки."}


@app.post("/predict", response_model=Response)
async def predict_sentiment(request: Request):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    encoded_input = tokenizer_vectors(request.question, padding=True, truncation=True, max_length=512,
                                      return_tensors='pt')
    with torch.no_grad():
        model_output = model_vectors(**encoded_input)

    embedding_tensor = mean_pooling(model_output, encoded_input['attention_mask'])
    embedding = embedding_tensor.squeeze().tolist()

    # Поисковые запросы
    search_chunk = query_cosine_similarity_chunks(embedding)
    pol_index_chunk = [item[1] for item in search_chunk]

    search_klasif1 = query_cosine_similarity_klasif1(embedding)
    pol_index_klasif1 = [item[1] for item in search_klasif1]

    search_klasif2 = query_cosine_similarity_klasif2(embedding)
    pol_index_klasif2 = [item[1] for item in search_klasif2]

    #Полиномная хеш-функция
    def polynomial_hash_combinations(array1, array2, array3):
        results = []
        k = 299
        all_text = ""
        klasif1 = None
        klasif2 = None
        textKlasif1 = ""
        textKlasif2 = ""

        for a, b, c in product(array1, array2, array3):
            hash_value = a + (b * k) + (c * k ** 2)
            results.append(hash_value)

        conn = psycopg2.connect("dbname='' user='' host='' password=''")
        cursor = conn.cursor()

        for hash_value in results:
            query = """
            SELECT polinom3, text_answer, index_klasif1, index_klasif2
            FROM answer_base
            WHERE polinom3 = %s
            LIMIT 5;
            """
            cursor.execute(query, (hash_value,))
            matches = cursor.fetchall()

            for i, row in enumerate(matches):
                polinom3, text_answer, index_Klasif1, index_Klasif2 = row
                all_text += text_answer + " "

                if i == 0 and klasif1 is None and klasif2 is None:
                    klasif1 = index_Klasif1
                    klasif2 = index_Klasif2

        if klasif1 is not None:
            cursor.execute("SELECT text_klasif1 FROM klasif1 WHERE index_klasif1 = %s", (klasif1,))
            result_klasif1 = cursor.fetchone()
            if result_klasif1:
                textKlasif1 = result_klasif1[0]

        if klasif2 is not None:
            cursor.execute("SELECT text_klasif2 FROM klasif2 WHERE index_klasif2 = %s", (klasif2,))
            result_klasif2 = cursor.fetchone()
            if result_klasif2:
                textKlasif2 = result_klasif2[0]

        cursor.close()
        conn.close()

        return [all_text, textKlasif1, textKlasif2]

    polinom_result = polynomial_hash_combinations(pol_index_chunk, pol_index_klasif1, pol_index_klasif2)

    if polinom_result[0] == '':
        search_index_for_llm = [item[1] for item in search_chunk]
        text_for_llm = search_no_polinom(search_index_for_llm)
        class1 = search_klasif1[0][0]
        class2 = search_klasif2[0][0]
    else:
        search_index_for_llm = [item[1] for item in search_chunk]
        text_for_llm = search_no_polinom(search_index_for_llm)
        print(text_for_llm)
        class1 = polinom_result[1]
        class2 = polinom_result[2]

    # Промпт
    messages = [
        {
            "role": "system",
            "content": (
                f"Ты вежливый и очень ответственный сотрудник технической поддержки российского видеохостинга 'Rutube'. "
                f"Твоя задача — отвечать на вопросы пользователей вежливо, кратко и строго по базе знаний. Вот база знаний: [{text_for_llm}]. "
                f"Отвечай только на вопрос пользователя, не добавляй ничего лишнего. ОТВЕТ МАКСИМУМ 2 ПРЕДЛОЖЕНИЯ. "
                f"Используй только ту информацию, которая содержится в базе знаний, и не добавляй ничего того, чего нет в базе знаний. "
                f"Если в базе знаний нет прямого ответа на вопрос, вежливо укажи на это, не придумывая ответов."
            )
        },
        {
            "role": "user",
            "content": f"{request.question}"
        }
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # Генерация ответа с семплированием и температурой
    outputs = model.generate(
        input_ids,
        max_new_tokens=350,
        eos_token_id=terminators,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.1,
        do_sample=True,
    )

    # Ответ llm
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Извлечение ответа после 'assistant' если оно есть в строчке ответа llm
    if "assistant" in full_response:
        answer = full_response.split("assistant")[-1].strip()
    else:
        answer = full_response

    # Возврат ответа с class_1 и class_2
    return Response(answer=answer, class_1=class1, class_2=class2)


# Запуск сервера
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
