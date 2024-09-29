import psycopg2


def connect_to_db():
    try:
        conn_params = "dbname='' user='' host='' password=''"
        conn = psycopg2.connect(conn_params)
        return conn
    except Exception as err:
        return f"Ошибка поключения {err}"

def query_cosine_similarity_chunks(embedding):
    conn = connect_to_db()
    cur = conn.cursor()
    embedding = str(embedding)
    try:
        query = f"""
        SELECT text_question, index_question, index_answer, 
               1 - (vectors <=> %s) AS cosine_similarity
        FROM questions_base
        ORDER BY cosine_similarity DESC
        LIMIT 5;
        """
        cur.execute(query, (embedding,))
        rows = cur.fetchall()
        return rows
    except Exception as err:
        print(err)
    finally:
        cur.close()
        conn.close()


def query_cosine_similarity_klasif1(embedding):
    conn = connect_to_db()
    cur = conn.cursor()
    embedding = str(embedding)
    try:
        query = f"""
        SELECT text_klasif1, index_klasif1, 
               1 - (vectors <=> %s) AS cosine_similarity
        FROM klasif1
        ORDER BY cosine_similarity DESC
        LIMIT 5;
        """
        cur.execute(query, (embedding,))
        rows = cur.fetchall()
        return rows
    except Exception as err:
        print(err)
    finally:
        cur.close()
        conn.close()

def query_cosine_similarity_klasif2(embedding):
    conn = connect_to_db()
    cur = conn.cursor()
    embedding = str(embedding)
    try:
        query = f"""
        SELECT text_klasif2, index_klasif2, 
               1 - (vectors <=> %s) AS cosine_similarity
        FROM klasif2
        ORDER BY cosine_similarity DESC
        LIMIT 5;
        """
        cur.execute(query, (embedding,))
        rows = cur.fetchall()
        return rows
    except Exception as err:
        print(err)
    finally:
        cur.close()
        conn.close()


def search_no_polinom(index_list):
    conn = connect_to_db()
    cur = conn.cursor()

    combined_text = []

    try:
        for index_value in index_list:
            cur.execute(
                """
                SELECT text_answer 
                FROM answer_base 
                WHERE index_answer = %s
                """,
                (index_value,)
            )

            result = cur.fetchone()

            if result:
                combined_text.append(result[0])

    except Exception as e:
        print(f"Ошибка при выполнении поиска: {e}")

    finally:
        cur.close()
        conn.close()
    return ' '.join(combined_text)



