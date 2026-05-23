"""
Contextus - Assistente Virtual para Dados Educacionais do IFBA (PNP)
Este sistema utiliza Inteligência Artificial (LLM) para transformar perguntas em
consultas a banco de dados (Text-to-SQL). Ele permite que gestores conversem com
os dados da Plataforma Nilo Peçanha sem precisarem terem conhecimento técnico
específico.
"""

import re
import json
import uuid
import groq
import pandas as pd
import streamlit as st
from pathlib import Path
from sqlalchemy import inspect
from langchain_groq import ChatGroq
from typing import Dict, Any, Union, List
from sqlalchemy import create_engine, Engine, text
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig, Runnable
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def build_schema_metadata(engine: Engine) -> str:
    """
    Constrói uma string com o esquema completo do banco, incluindo as descrições
    oficiais das colunas obtidas das tabelas '*DicionarioDados'.
    """
    inspector = inspect(engine)
    all_tables = inspector.get_table_names()

    # Separa as tabelas de dados das de dicionário
    data_tables = [t for t in all_tables if not t.endswith("DicionarioDados")]
    dict_tables = [t for t in all_tables if t.endswith("DicionarioDados")]

    metadata_str = ""

    for table in data_tables:
        columns_info = inspector.get_columns(table)
        columns = [col["name"] for col in columns_info]
        metadata_str += f"\nTabela: {table}\nColunas: {', '.join(columns)}\n"

        # Procura o dicionário correspondente (ex: EficienciaAcademica -> EficienciaAcademicaDicionarioDados)
        dict_name = f"{table}DicionarioDados"
        if dict_name in dict_tables:
            try:
                dict_df = pd.read_sql(f'SELECT * FROM "{dict_name}"', engine)
                # Ajuste conforme a estrutura real do seu dicionário. Exemplo:
                # Assume colunas: 'Campo', 'Descrição'
                if "Campo" in dict_df.columns and "Descrição" in dict_df.columns:
                    for _, row in dict_df.iterrows():
                        metadata_str += f"  - {row['Campo']}: {row['Descrição']}\n"
                else:
                    # Fallback: imprime as primeiras linhas
                    metadata_str += f"  (Dicionário sem colunas esperadas. Primeiros registros):\n{dict_df.head().to_string()}\n"
            except Exception:
                metadata_str += "  (Não foi possível ler a tabela de dicionário)\n"

    return metadata_str


class SQLChatMessageHistory(BaseChatMessageHistory):
    """
    Gerenciador de Memória das Conversas.
    Esta classe atua como o 'cérebro' de longo prazo do assistente, salvando e
    recuperando tudo o que o usuário e a IA disseram para que o robô não perca
    o contexto durante o bate-papo.
    """

    def __init__(self, session_id: str, db_path: str = "sqlite:///database.db") -> None:
        self.session_id = session_id
        # Cria uma conexão com o banco de dados SQLite local
        self.engine = create_engine(db_path)
        self._ensure_table()

    def _ensure_table(self):
        """Garante que a tabela de histórico exista antes de tentar ler ou gravar."""
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Busca todas as mensagens de uma sessão específica e as recria como objetos Langchain."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT role, content FROM chat_messages WHERE session_id = :sid ORDER BY timestamp"
                ),
                {"sid": self.session_id},
            )
            messages = []
            for row in result:
                if row.role == "human":
                    messages.append(HumanMessage(content=row.content))
                elif row.role == "ai":
                    messages.append(AIMessage(content=row.content))
            return messages

    def add_message(self, message: BaseMessage) -> None:
        """Salva uma nova mensagem (seja do humano ou da IA) no banco de dados."""
        role = "human" if isinstance(message, HumanMessage) else "ai"
        with self.engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO chat_messages (session_id, role, content) VALUES (:sid, :role, :content)"
                ),
                {"sid": self.session_id, "role": role, "content": message.content},
            )
            conn.commit()

    def clear(self) -> None:
        """Apaga o histórico de uma conversa específica."""
        with self.engine.connect() as conn:
            conn.execute(
                text("DELETE FROM chat_messages WHERE session_id = :sid"),
                {"sid": self.session_id},
            )
            conn.commit()


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Função auxiliar exigida pelo Langchain para resgatar a memória baseada no ID da sessão."""
    return SQLChatMessageHistory(session_id=session_id)


def load_all_sessions(db_path: str = "sqlite:///database.db") -> List[str]:
    """
    Carrega o identificador de todas as conversas anteriores.
    Isso permite que o usuário veja no menu lateral seu histórico de chats antigos.
    """
    engine = create_engine(db_path)
    with engine.connect() as conn:
        # Tenta criar a tabela caso o app seja iniciado e a primeira ação seja ler sessões
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()

        # Agrupa pelo ID da sessão, ordenando da conversa mais recente para a mais antiga
        result = conn.execute(text("""
            SELECT session_id 
            FROM chat_messages 
            GROUP BY session_id 
            ORDER BY MAX(timestamp) DESC
        """))
        return [row[0] for row in result]


def delete_session_from_db(session_id: str, db_path: str = "sqlite:///database.db"):
    """Exclui permanentemente uma conversa específica do banco de dados."""
    engine = create_engine(db_path)
    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM chat_messages WHERE session_id = :sid"),
            {"sid": session_id},
        )
        conn.commit()


def load_csvs_to_sqlite(docs_dir: Union[Path, str], db_path: str) -> Engine:
    """
    Mecanismo de Ingestão de Dados.
    Lê todas as planilhas (arquivos CSV) contendo os dados do IFBA e as transforma
    em tabelas dentro de um banco de dados relacional. É isso que permite a IA
    realizar análises estatísticas em tempo real.
    """
    engine = create_engine(url=db_path)
    csv_files = list(Path(docs_dir).glob("*.csv"))

    if not csv_files:
        return engine

    for csv_path in csv_files:
        table_name = Path(csv_path).stem  # O nome da tabela será o nome do arquivo
        try:
            # Tenta ler com separador de ponto e vírgula (padrão de muitos sistemas governamentais)
            df = pd.read_csv(filepath_or_buffer=csv_path, sep=";")
            if len(df.columns) <= 1:
                # Fallback para vírgula caso a leitura acima falhe silenciosamente (retornando só 1 coluna)
                df = pd.read_csv(filepath_or_buffer=csv_path, sep=",")
        except Exception:
            # Fallback definitivo para separador vírgula
            df = pd.read_csv(filepath_or_buffer=csv_path, sep=",")

        # Se a tabela já existir, ele a substitui (if_exists="replace")
        df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)

    return engine


def create_sql_agent_with_db(db_path: str) -> Runnable[Any, Any]:
    """
    Cria o agente Text-to-SQL completo, pronto para ser executado com histórico de conversas.

    Etapas:
      1. Conecta ao banco SQLite gerado a partir dos CSVs.
      2. Constrói o esquema (com descrições das colunas) para orientar o LLM.
      3. Define o modelo de linguagem (LLM) da Groq.
      4. Cria a cadeia de geração de SQL (prompt -> LLM -> parser -> limpeza).
      5. Cria o prompt de resposta final (texto + definição de gráfico).
      6. Define a função de execução do SQL contra o banco.
      7. Encadeia tudo usando RunnablePassthrough.
      8. Envolve a cadeia com RunnableWithMessageHistory para manter contexto da conversa.
    """

    # ----- 1. Conexão com o banco de dados -----
    engine = create_engine(db_path)
    db = SQLDatabase(engine=engine)

    # ----- 2. Metadata do esquema (nomes de tabelas, colunas e descrições) -----
    metadata = build_schema_metadata(engine)

    # ----- 3. Modelo de linguagem (LLM) -----
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.0,  # Zero para respostas determinísticas
        api_key=st.secrets["groq_api_key"],
    )

    # ----- 4. Prompt e cadeia de geração de SQL -----
    sql_generation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
                    Você é um gerador de consultas SQL para um banco SQLite com dados de evasão do Campus Jacobina.
                    Com base no esquema abaixo, crie apenas a consulta SQL que responderá à pergunta do usuário.
                    **Retorne exclusivamente a instrução SQL pura, sem markdown, nem explicações.**

                    Regras:
                        - Nas tabelas de dados (que possuem a coluna `nomeUnidadeRecente`), inclua obrigatoriamente `WHERE nomeUnidadeRecente = 'Campus Jacobina'`.
                        - Se a pergunta pedir a definição de um termo, gere um `SELECT` na tabela de dicionário correspondente (ex.: `SELECT Descrição FROM ... WHERE Campo = '...'`).
                        - Use apenas `SELECT`; nunca `INSERT`, `UPDATE`, `DELETE`, `DROP`.
                        - Envolva com aspas duplas todo nome de coluna ou tabela que contenha espaços, |, %, parênteses ou outros caracteres especiais.
                        - Se a pergunta for inviável com os dados disponíveis, retorne `'N/A'`.
                        - Sempre envolva com aspas duplas os nomes de tabelas ou colunas que contenham espaços ou caracteres especiais (ex.: `"EficienciaAcademica DicionarioDados"`).
                        - Substitua qualquer ocorrência do texto literal “N/A” por “informação indisponível” ou equivalente, nunca exiba “N/A”.

                Esquema do banco:
                {metadata}
                """,
            ),
            MessagesPlaceholder(variable_name="chat_history"),  # Histórico da conversa
            ("human", "{input}"),  # Pergunta atual
        ]
    )

    # Encadeia: prompt -> LLM -> extração de string -> limpeza de markdown
    generate_sql = (
        sql_generation_prompt
        | llm
        | StrOutputParser()
        | (lambda sql: sql.strip().replace("```sql", "").replace("```", ""))
    )

    # ----- 5. Prompt final para a resposta contextualizada (com gráficos) -----
    final_response_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Você é Contextus, assistente virtual especializado nos dados de **evasão** do Campus Jacobina do IFBA, extraídos da Plataforma Nilo Peçanha (PNP). **Comunique-se sempre em português**; se o usuário usar outro idioma, avise educadamente.

                **Suas capacidades:**

                - **Interpretação em Linguagem Natural:** Compreender perguntas feitas em português sobre os indicadores educacionais de evasão do IFBA - Campus Jacobina.
                - **Consulta de Dados (Text-to-SQL):** Transformar perguntas em consultas SQL para buscar dados precisos na base da PNP.
                - **Respostas Baseadas em Fatos:** Basear-se exclusivamente nos resultados obtidos da base de dados, sem nunca inventar ou alucinar informações.
                - **Geração de Gráficos:** Criar visualizações de dados dinâmicas em formato de gráficos de linha (para séries temporais) ou de barra (para comparações) quando apropriado.
                - **Memória de Contexto:** Lembrar do histórico da conversa para manter a continuidade e o fluxo lógico do diálogo.
                - **Foco e Restrição de Escopo:** Limitar as respostas apenas ao contexto da evasão no Campus Jacobina, recusando educadamente perguntas sobre outros temas, outros campi ou outras instituições.

                **Diretrizes de resposta:**

                - Baseie-se **exclusivamente** no resultado SQL fornecido. Nunca invente dados.
                - Se a pergunta for uma saudação ou pedir informações sobre suas capacidades, responda amigavelmente explicando o escopo, **sem depender dos dados**.
                - Sempre responda utilizando português do Brasil.
                - Para perguntas fora do escopo (outros campi, IFs, temas não relacionados à evasão do Campus Jacobina), recuse educadamente.
                - Quando o resultado for `'N/A'`, vazio ou indicar erro, informe que a informação não foi encontrada e sugira reformular a pergunta. **Nunca exiba mensagens técnicas de erro**.
                - **Nunca** mostre a consulta SQL na resposta, apenas os resultados interpretados.
                - Substitua qualquer ocorrência do texto literal “N/A” por “informação indisponível” ou equivalente, nunca exiba “N/A”.
                - Valores percentuais estão como fração (ex.: 0,09 = 9%). Multiplique por 100 e exiba com o símbolo %.
                - Se um dado numérico estiver vazio ou NULL, indique que a informação não está disponível para esse recorte, sem supor zero.

                **Gráfico (apenas se houver dados comparativos ou série histórica):**
                Inclua ao final um bloco JSON válido, entre crases (```json ... ```), com a seguinte estrutura:

                ```json
                {{
                    "chart_type": "bar",
                    "data": [
                        {{"x": "Nome da Categoria ou Ano", "y": 42}},
                        {{"x": "Outra Categoria", "y": 18}}
                    ]
                }}
                ```

                Regras do gráfico:

                - `chart_type` = `"line"` para séries temporais, `"bar"` para comparações.
                - **Limite os dados a no máximo 12 itens**; se houver mais, agrupe ou mostre apenas os principais.
                - Mantenha `x` como string descritiva e `y` como número.
                - Não inclua tabelas markdown com os mesmos dados já representados no gráfico.
                - `x` sempre string, `y` número; o JSON precisa ser estritamente válido (sem vírgulas finais).
                """,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "Pergunta: {input}\nSQL gerada: {query}\nResultado: {raw_result}",
            ),
        ]
    )

    # ----- 6. Função de execução do SQL gerado -----
    def execute_sql(sql: str) -> str:
        """Executa a consulta SQL e retorna o resultado como string."""
        if sql.upper() == "N/A" or not sql:
            return "N/A"
        try:
            result = db.run(sql)
            return str(result) if result is not None else "N/A"
        except Exception as e:
            return f"Erro na consulta: {e}"

    # ----- 7. Construção da cadeia principal (pipeline) -----
    chain = (
        # Passo 1: gera a consulta SQL e adiciona ao dicionário "query"
        RunnablePassthrough.assign(query=generate_sql)
        # Passo 2: executa o SQL e adiciona o resultado em "raw_result"
        | RunnablePassthrough.assign(raw_result=lambda x: execute_sql(x["query"]))
        # Passo 3: formata a resposta final (texto + JSON de gráfico) usando o LLM
        | RunnablePassthrough.assign(
            output=lambda x: (final_response_prompt | llm | StrOutputParser()).invoke(
                {
                    "input": x["input"],
                    "query": x["query"],
                    "raw_result": x["raw_result"],
                    "chat_history": x["chat_history"],
                }
            )
        )
        # Passo 4: extrai apenas o campo "output" para a resposta final
        | (lambda x: {"output": x["output"]})
    )

    # ----- 8. Agente com histórico de conversa (memória) -----
    agent_with_history = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_session_history,  # Função que resgata a memória da sessão
        input_messages_key="input",  # Chave de entrada da mensagem do usuário
        history_messages_key="chat_history",  # Chave onde o histórico será injetado
    )

    return agent_with_history


@st.cache_resource
def init_agent() -> Runnable[Any, Any]:
    """
    Inicializa o agente apenas uma vez (cache).
    Garante que a pasta de documentos exista e que os dados sejam carregados
    no banco de dados antes da aplicação web iniciar de fato.
    """
    Path("docs").mkdir(exist_ok=True)
    load_csvs_to_sqlite(docs_dir="docs", db_path="sqlite:///database.db")
    return create_sql_agent_with_db(db_path="sqlite:///database.db")


def load_messages_for_ui(session_id: str) -> List[Dict[str, str]]:
    """Transforma o histórico do banco de dados em um formato que a tela do aplicativo consiga desenhar."""
    history = SQLChatMessageHistory(session_id=session_id)
    ui_messages = []
    for msg in history.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        ui_messages.append({"role": role, "content": msg.content})
    return ui_messages


def render_message_with_charts(content: str):
    """
    Mecanismo de Visualização de Dados.
    Verifica se a resposta da IA contém apenas texto ou se a IA pediu para
    desenhar um gráfico (usando a estrutura JSON definida no prompt). Se houver
    pedido de gráfico, ele é renderizado dinamicamente na tela.
    """
    # Regex para capturar blocos markdown de JSON escondidos na resposta da IA
    json_pattern = re.compile(r"```json\n(.*?)\n```", re.DOTALL)
    parts = json_pattern.split(content)

    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Texto normal de explicação para o usuário
            if part.strip():
                st.markdown(part)
        else:
            # Bloco JSON detectado: tenta convertê-lo em um gráfico visual
            try:
                data = json.loads(part)
                if "chart_type" in data and "data" in data:
                    df = pd.DataFrame(data["data"])

                    if not df.empty and "x" in df.columns and "y" in df.columns:
                        # O Streamlit requer que a coluna 'x' seja o índice para renderizar corretamente
                        df.set_index("x", inplace=True)

                        chart_type = data.get("chart_type", "bar")

                        # Decide entre gráfico de linha (séries históricas) ou de barras (comparações)
                        if chart_type == "line":
                            st.line_chart(df)
                        else:
                            st.bar_chart(df)
                    else:
                        # Se os dados estiverem malformados, exibe o conteúdo cru para não ocultar a informação
                        st.json(data)
                else:
                    # Se não for um JSON formatado para gráfico, exibe como bloco de código comum
                    st.code(part, language="json")
            except json.JSONDecodeError:
                # Tratamento de erro: Se o LLM alucinar a sintaxe do JSON, exibe como texto seguro
                st.code(part, language="json")


def main() -> None:
    """
    Ponto de Entrada da Interface de Usuário (Frontend).
    Controla a tela principal, a barra lateral de histórico, e gerencia
    o envio e recebimento de mensagens através do Streamlit.
    """
    # Configuração básica da aba do navegador
    st.set_page_config(page_title="Contextus", page_icon="🤖")

    # ----- GERENCIAMENTO DE ESTADO (SESSION STATE) -----
    # O Streamlit recarrega a página inteira a cada interação. O session_state é usado
    # para guardar as variáveis para que elas não sejam perdidas durante o recarregamento.

    if "chat_sessions" not in st.session_state:
        existing_sessions = load_all_sessions()
        st.session_state["chat_sessions"] = (
            existing_sessions if existing_sessions else []
        )

    if "all_messages" not in st.session_state:
        st.session_state["all_messages"] = {}

    # Define qual é a conversa atual. Se não houver, cria uma nova com um ID aleatório.
    if "session_id" not in st.session_state:
        if st.session_state["chat_sessions"]:
            first_session = st.session_state["chat_sessions"][0]
        else:
            first_session = f"sessao_{uuid.uuid4().hex[:8]}"
            st.session_state["chat_sessions"].append(first_session)
        st.session_state["session_id"] = first_session
        st.session_state["all_messages"][first_session] = load_messages_for_ui(
            first_session
        )

    current_session = st.session_state["session_id"]
    if current_session not in st.session_state["all_messages"]:
        st.session_state["all_messages"][current_session] = load_messages_for_ui(
            current_session
        )

    # ----- CABEÇALHO DA INTERFACE -----
    col_logo, col_title = st.columns(
        [0.15, 0.85], gap="small", vertical_alignment="center"
    )
    st.info(
        "**Precisa de ajuda?** Consulte a [documentação oficial do Contextus]"
        "(https://zolppy.github.io/contextus) para definições, escopo e exemplos de uso.",
        icon="📘",
    )
    with col_logo:
        st.image("assets/logo.jpg", width=120)
    with col_title:
        st.header("💬 :green[Contextus (v1.0.7)]")
        st.subheader(
            "Agente especializado em dados de evasão do Campus Jacobina, disponíveis na Plataforma Nilo Peçanha."
        )

    MAX_SESSIONS = 6

    # ----- BARRA LATERAL (SIDEBAR) -----
    with st.sidebar:
        # Lógica de bloqueio para não sobrecarregar o banco com infinitas conversas
        if len(st.session_state["chat_sessions"]) >= MAX_SESSIONS:
            st.button(
                label="Nova conversa",
                use_container_width=True,
                type="primary",
                icon="✉️",
                disabled=True,
                help=f"Limite de {MAX_SESSIONS} conversas atingido. Exclua uma conversa para criar uma nova.",
            )
        else:
            if st.button(
                label="Nova conversa",
                use_container_width=True,
                type="primary",
                icon="✉️",
            ):
                # Cria e foca em uma nova sessão vazia
                new_session_id = f"sessao_{uuid.uuid4().hex[:8]}"
                st.session_state["chat_sessions"].insert(0, new_session_id)
                st.session_state["session_id"] = new_session_id
                st.session_state["all_messages"][new_session_id] = []
                st.rerun()

        st.divider()
        st.subheader("Histórico de Conversas")

        def delete_session(session_to_delete: str):
            """Função local na UI para processar a remoção de chats no banco e na tela."""
            delete_session_from_db(session_to_delete)
            st.session_state["chat_sessions"] = [
                s for s in st.session_state["chat_sessions"] if s != session_to_delete
            ]
            if session_to_delete in st.session_state["all_messages"]:
                del st.session_state["all_messages"][session_to_delete]

            # Se excluiu a conversa que estava aberta, redireciona para a próxima ou cria uma nova
            if st.session_state["session_id"] == session_to_delete:
                if st.session_state["chat_sessions"]:
                    st.session_state["session_id"] = st.session_state["chat_sessions"][
                        0
                    ]
                else:
                    new_id = f"sessao_{uuid.uuid4().hex[:8]}"
                    st.session_state["chat_sessions"].append(new_id)
                    st.session_state["session_id"] = new_id
                    st.session_state["all_messages"][new_id] = []

        # Renderiza os botões das conversas anteriores (Limitado dinamicamente pelas regras acima)
        sessions_to_display = st.session_state["chat_sessions"].copy()
        for sess_id in sessions_to_display:
            if sess_id not in st.session_state["all_messages"]:
                st.session_state["all_messages"][sess_id] = load_messages_for_ui(
                    sess_id
                )

            mensagens_sessao = st.session_state["all_messages"].get(sess_id, [])

            # O nome do botão no menu será um trecho da primeira pergunta feita
            titulo_botao = (
                mensagens_sessao[0]["content"][:25] + "..."
                if mensagens_sessao
                else "Conversa atual"
            )

            # Destaca visualmente a conversa que o usuário está olhando agora
            is_active = sess_id == st.session_state["session_id"]
            button_type = "primary" if is_active else "secondary"

            col1, col2 = st.columns([8, 1])
            with col1:
                # Botão para alternar entre as conversas
                if st.button(
                    label=titulo_botao,
                    key=f"select_{sess_id}",
                    use_container_width=True,
                    type=button_type,
                ):
                    st.session_state["session_id"] = sess_id
                    st.rerun()
            with col2:
                # Botão de lixeira
                if st.button(
                    label="",
                    key=f"delete_{sess_id}",
                    help="Excluir conversa",
                    use_container_width=True,
                    icon="🗑️",
                ):
                    delete_session(sess_id)
                    st.rerun()

    # Inicializa ou resgata a conexão com o Agente de Inteligência Artificial
    agent = init_agent()

    # Variável de atalho para acessar a lista de mensagens atuais
    current_messages = st.session_state["all_messages"][st.session_state["session_id"]]

    # ----- RENDERIZAÇÃO DO CHAT -----
    # Desenha na tela todas as mensagens trocadas até o momento nesta conversa
    for msg in current_messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_message_with_charts(msg["content"])
            else:
                st.markdown(msg["content"])

    # ----- CAMPO DE ENTRADA DO USUÁRIO -----
    user_input = st.chat_input("Envie uma mensagem", max_chars=1000)

    if user_input:
        # 1. Exibe e salva a mensagem do humano
        current_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2. Processa e exibe a resposta da IA
        with st.chat_message("assistant"):
            with st.spinner("Analisando dados, por favor, aguarde."):
                try:
                    # Passa o ID da sessão para que a IA lembre do contexto (perguntas anteriores)
                    config: RunnableConfig = {
                        "configurable": {"session_id": st.session_state["session_id"]}
                    }

                    # Envia a pergunta para o modelo executar a lógica Text-to-SQL
                    resposta_dict = agent.invoke({"input": user_input}, config=config)
                    resposta = str(resposta_dict["output"])

                    # Processa a resposta (exibindo texto ou desenhando os gráficos)
                    render_message_with_charts(resposta)

                    # Salva a resposta no histórico da interface
                    current_messages.append({"role": "assistant", "content": resposta})

                    # Sincroniza a memória da tela com a memória do banco de dados
                    st.session_state["all_messages"][st.session_state["session_id"]] = (
                        load_messages_for_ui(st.session_state["session_id"])
                    )

                # Tratamentos amigáveis de erros da API, úteis para não assustar usuários não-técnicos
                except groq.APITimeoutError:
                    st.error("Tempo esgotado. Tente novamente.")
                except groq.APIConnectionError:
                    st.error(
                        "Não foi possível se conectar à API. Verifique sua internet."
                    )
                except groq.InternalServerError:
                    st.error(
                        "Erro interno no servidor da Groq. Aguarde e tente novamente."
                    )
                except groq.RateLimitError:
                    st.error(
                        "Limite de requisições por minuto atingido. Aguarde um pouco e tente novamente."
                    )
                except groq.APIStatusError as e:
                    if e.status_code == 413:
                        st.error(
                            "Sua pergunta gerou um contexto muito grande para o modelo atual. Por favor, tente novamente."
                        )
                except Exception:
                    st.error(
                        "Ocorreu um erro inesperado. Por favor, tente novamente mais tarde."
                    )


if __name__ == "__main__":
    main()
