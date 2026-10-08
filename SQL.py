"""
Text-to-SQL
Converts natural language questions into SQL queries using the Claude API,
then runs them against a small sample SQLite database and shows the results.

Requires: pip install anthropic
Requires an Anthropic API key: https://console.anthropic.com/settings/keys
Set it as an environment variable before running:
  Windows (PowerShell):  $env:ANTHROPIC_API_KEY="your-key-here"
  macOS/Linux:           export ANTHROPIC_API_KEY="your-key-here"
"""

import os
import sqlite3
import re
from anthropic import Anthropic

MODEL = "claude-sonnet-5"
DB_PATH = "sample_store.db"


def setup_database():
    """Creates a small sample SQLite database with employees and sales tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS employees")
    cursor.execute("DROP TABLE IF EXISTS sales")

    cursor.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            salary INTEGER,
            hire_year INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            employee_id INTEGER,
            product TEXT,
            amount INTEGER,
            sale_date TEXT
        )
    """)

    employees = [
        (1, "Alice Johnson", "Sales", 55000, 2020),
        (2, "Bob Smith", "Engineering", 75000, 2019),
        (3, "Carol Davis", "Sales", 60000, 2021),
        (4, "David Lee", "Marketing", 58000, 2022),
        (5, "Emma Wilson", "Engineering", 82000, 2018),
    ]
    cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?)", employees)

    sales = [
        (1, 1, "Laptop", 1200, "2026-01-15"),
        (2, 1, "Monitor", 300, "2026-02-10"),
        (3, 3, "Laptop", 1200, "2026-01-20"),
        (4, 3, "Keyboard", 80, "2026-03-05"),
        (5, 1, "Mouse", 25, "2026-03-12"),
    ]
    cursor.executemany("INSERT INTO sales VALUES (?, ?, ?, ?, ?)", sales)

    conn.commit()
    conn.close()
    print(f"Sample database created: {DB_PATH} (tables: employees, sales)")


def get_schema_description():
    return """
Table: employees
  - id (INTEGER, primary key)
  - name (TEXT)
  - department (TEXT)
  - salary (INTEGER)
  - hire_year (INTEGER)

Table: sales
  - id (INTEGER, primary key)
  - employee_id (INTEGER, references employees.id)
  - product (TEXT)
  - amount (INTEGER)
  - sale_date (TEXT, format YYYY-MM-DD)
"""


def get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print("Get a key at https://console.anthropic.com/settings/keys")
        exit(1)
    return Anthropic(api_key=api_key)


def generate_sql(client, question, schema):
    prompt = f"""You are a SQL expert. Given this database schema:

{schema}

Convert the following question into a single valid SQLite SQL query.
Only output the SQL query itself, nothing else - no explanation, no markdown formatting, no backticks.

Question: {question}
SQL query:"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    sql = "".join(block.text for block in response.content if block.type == "text")
    sql = sql.strip()
    # Remove markdown code fences if the model added them anyway
    sql = re.sub(r"^```sql\s*|^```\s*|```$", "", sql, flags=re.MULTILINE).strip()
    return sql


def run_query(sql):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        columns = [description[0] for description in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return columns, rows, None
    except Exception as e:
        return None, None, str(e)
    finally:
        conn.close()


def print_results(columns, rows):
    if not rows:
        print("(No results)")
        return

    header = " | ".join(columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join(str(v) for v in row))


def main():
    print("=" * 50)
    print("Text-to-SQL")
    print("=" * 50)

    setup_database()
    schema = get_schema_description()
    print("\nSchema:")
    print(schema)

    client = get_client()

    print("Ask questions in plain English, e.g.:")
    print('  "What is the average salary in the Engineering department?"')
    print('  "Who sold the most laptops?"')
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye! 👋")
            break
        if not question:
            continue

        try:
            sql = generate_sql(client, question, schema)
        except Exception as e:
            print(f"Error calling the API: {e}")
            continue

        print(f"\nGenerated SQL:\n  {sql}\n")

        columns, rows, error = run_query(sql)
        if error:
            print(f"Error running query: {error}")
        else:
            print_results(columns, rows)
        print()


if __name__ == "__main__":
    main()