import psycopg2
from psycopg2 import sql

def connect_to_db():
    try:
        conn_params = "dbname='' user='' host='' password=''"
        conn = psycopg2.connect(conn_params)
        return conn
    except Exception as err:
        return f"Ошибка поключения {err}"


##Код для создания таблиц по нашему кейсу

# def create_vector_table():
#     conn = connect_to_db()
#     cur = conn.cursor()
#     create_table_query = """
#     CREATE TABLE IF NOT EXISTS questions_base(
#         id bigserial PRIMARY KEY,
#         text_question text,
#         vectors vector(1024),
#         index_question integer,
#         index_answer integer
#     );
#     """  # Здесь добавлено уникальное ограничение для колонки groupirovka
#     cur.execute(create_table_query)
#     conn.commit()  # Зафиксируем создание таблицы в базе данных
#     cur.close()
#     conn.close()
#
# create_vector_table()


# def create_vector_table():
#     conn = connect_to_db()
#     cur = conn.cursor()
#     create_table_query = """
#     CREATE TABLE IF NOT EXISTS answer_base(
#         id bigserial PRIMARY KEY,
#         name_answer text,
#         polinom3 integer,
#         polinom2 integer,
#         index_answer integer,
#         index_question integer,
#         index_Klasif1 integer,
#         index_Klasif2 integer
#     );
#     """  # Здесь добавлено уникальное ограничение для колонки groupirovka
#     cur.execute(create_table_query)
#     conn.commit()  # Зафиксируем создание таблицы в базе данных
#     cur.close()
#     conn.close()
#
# create_vector_table()
#
#
#
# def create_vector_table():
#     conn = connect_to_db()
#     cur = conn.cursor()
#     create_table_query = """
#     CREATE TABLE IF NOT EXISTS Klasif1(
#         id bigserial PRIMARY KEY,
#         text_klasif1 text,
#         vectors vector(1024),
#         index_Klasif1 integer
#     );
#     """  # Здесь добавлено уникальное ограничение для колонки groupirovka
#     cur.execute(create_table_query)
#     conn.commit()  # Зафиксируем создание таблицы в базе данных
#     cur.close()
#     conn.close()
#
# create_vector_table()
#
#
# def create_vector_table():
#     conn = connect_to_db()
#     cur = conn.cursor()
#     create_table_query = """
#     CREATE TABLE IF NOT EXISTS Klasif2(
#         id bigserial PRIMARY KEY,
#         text_klasif2 text,
#         vectors vector(1024),
#         index_Klasif2 integer
#     );
#     """  # Здесь добавлено уникальное ограничение для колонки groupirovka
#     cur.execute(create_table_query)
#     conn.commit()  # Зафиксируем создание таблицы в базе данных
#     cur.close()
#     conn.close()
#
# create_vector_table()