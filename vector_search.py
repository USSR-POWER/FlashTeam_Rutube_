from transformers import AutoTokenizer, AutoModel
import torch
from search_func import query_cosine_similarity_chunks, query_cosine_similarity_klasif1, query_cosine_similarity_klasif2, search_no_polinom
from itertools import product
import psycopg2

tokenizer_vectors = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
model_vectors = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")


question = 'Как сменить пароль' #Введите свой запрос


#Получение вектора
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Первый элемент содержит эмбеддинги токенов
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


encoded_input = tokenizer_vectors(question, padding=True, truncation=True, max_length=512, return_tensors='pt')
with torch.no_grad():
    model_output = model_vectors(**encoded_input)

embedding_tensor = mean_pooling(model_output, encoded_input['attention_mask'])
embedding = embedding_tensor.squeeze().tolist()

#Поисковые запросы
search_chunk = query_cosine_similarity_chunks(embedding)
pol_index_chunk = [item[1] for item in search_chunk]

search_klasif1 = query_cosine_similarity_klasif1(embedding)
pol_index_klasif1 = [item[1] for item in search_klasif1]

search_klasif2 = query_cosine_similarity_klasif2(embedding)
pol_index_klasif2 = [item[1] for item in search_klasif2]


def polynomial_hash_combinations(array1, array2, array3):
    """
    Функция для вычисления полиномиальных хеш-значений для всех комбинаций трех массивов.
    Выполняет поиск совпадений в таблице PostgreSQL и собирает данные.

    :param array1: Первый массив с ID свойств (по убыванию косинусного сходства)
    :param array2: Второй массив с ID свойств (по убыванию косинусного сходства)
    :param array3: Третий массив с ID свойств (по убыванию косинусного сходства)
    :param k: Фиксированное число, используемое в полиномиальной хеш-функции
    :return: Массив с тремя элементами [all_text, textKlasif1, textKlasif2]
    """
    # Массив для хранения результатов полиномиальных хеш-значений
    results = []

    # Переменные для текстового результата и классификаций
    k = 299
    all_text = ""
    klasif1 = None
    klasif2 = None
    textKlasif1 = ""
    textKlasif2 = ""

    # Генерация всех возможных комбинаций элементов из трех массивов
    for a, b, c in product(array1, array2, array3):
        # Вычисление полиномиального хеш-значения для комбинации
        hash_value = a + (b * k) + (c * k ** 2)
        results.append(hash_value)
    print(results)

    # Подключение к базе данных PostgreSQL
    conn = psycopg2.connect("dbname='' user='' host='' password=''")
    cursor = conn.cursor()

    # Перебор значений массива results и поиск совпадений в таблице
    for hash_value in results:
        # Выполнение запроса для поиска 5 первых совпадений
        query = """
        SELECT polinom3, text_answer, index_klasif1, index_klasif2
        FROM answer_base
        WHERE polinom3 = %s
        LIMIT 5;
        """
        cursor.execute(query, (hash_value,))
        matches = cursor.fetchall()

        # Обработка найденных совпадений
        for i, row in enumerate(matches):
            polinom3, text_answer, index_Klasif1, index_Klasif2 = row
            # Добавление текста ответа в all_text
            all_text += text_answer + " "

            # Присвоение klasif1 и klasif2 при первом совпадении
            if i == 0 and klasif1 is None and klasif2 is None:
                klasif1 = index_Klasif1
                klasif2 = index_Klasif2

    # Поиск значений text_klasif1 и text_klasif2, если klasif1 и klasif2 найдены
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

    # Закрытие соединения с базой данных
    cursor.close()
    conn.close()

    # Возвращаем массив с результатами
    return [all_text, textKlasif1, textKlasif2]


polinom_result = polynomial_hash_combinations(pol_index_chunk, pol_index_klasif1, pol_index_klasif2)


if polinom_result[0] == '':
    search_index_for_llm = [item[1] for item in search_chunk]
    text_for_llm = search_no_polinom(search_index_for_llm)
    print(text_for_llm)
    class1 = search_klasif1[0][0]
    print(class1)
    class2 = search_klasif2[0][0]
    print(class2)
else:
    text_for_llm = polinom_result[0]
    print(text_for_llm)
    class1 = polinom_result[1]
    print(class1)
    class2 = polinom_result[2]
    print(class2)



