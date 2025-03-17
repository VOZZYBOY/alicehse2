import os
import sys
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import telebot
from telebot import types


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Параметры из .env
load_dotenv()
GIGACHAT_API_KEY = "OTkyYTgyNGYtMjRlNC00MWYyLTg3M2UtYWRkYWVhM2QxNTM1OjI5YWY1YWFlLWVmMzItNDE1MS1iY2YwLWYxODAyOTZmMmUwNg=="
GIGACHAT_SCOPE = os.getenv('GIGACHAT_SCOPE', 'GIGACHAT_API_PERS')
KNOWLEDGE_BASE_PATH = os.getenv('KNOWLEDGE_BASE_PATH', 'knowledge_base.txt')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')

class RagChatWithMemory:
    def __init__(self):
        print("Инициализация чат-бота с памятью...")
        
        
        self.chat_model = GigaChat(
            credentials=GIGACHAT_API_KEY,
            scope=GIGACHAT_SCOPE,
            verify_ssl_certs=False,
            model="GigaChat"
        )
        
     
        self.embedding_model = GigaChatEmbeddings(
            credentials=GIGACHAT_API_KEY,
            scope=GIGACHAT_SCOPE,
            model="Embeddings",
            verify_ssl_certs=False
        )
        
       
        self.knowledge_base = self.load_knowledge_base(KNOWLEDGE_BASE_PATH)
        if not self.knowledge_base:
            sys.exit(1)
        self.embeddings = self.create_embeddings(self.knowledge_base)
        if not self.embeddings:
            sys.exit(1)
            
        
        self.conversation_history = {}
        self.system_message = SystemMessage(content="""
        Ты — полезный ассистент для студентов школы дизайна. Ты отвечаешь на вопросы о правилах поведения в школе.
        Ты умеешь находить информацию в базе знаний и предоставлять точные ответы.
        
        Используй предоставленный контекст для ответа на вопрос. Если в контексте нет прямого ответа, 
        сообщи об этом пользователю, но постарайся быть полезным. Не выдумывай информацию.
        
        Анализируй историю диалога для понимания контекста вопроса. Если в новом вопросе не хватает 
        деталей, но они упоминались ранее в диалоге, используй их для формирования ответа.
        
        При ответе на вопросы о времени, расписании или других конкретных деталях,
        всегда давай точную информацию из базы знаний. Например, на вопрос "Во сколько начинается вторая пара?",
        нужно ответить с конкретным временем из расписания звонков.
        """)

    def load_knowledge_base(self, filepath: str) -> List[str]:
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            fragments = content.split('\\\\')
            fragments = [f.strip() for f in fragments if f.strip()]
            print(f"Загружено {len(fragments)} фрагментов из базы знаний")
            return fragments
        except Exception as e:
            print(f"Ошибка при загрузке базы знаний: {e}")
            return []
    
    def create_embeddings(self, fragments: List[str]) -> Dict[str, List[float]]:
       
        try:
            documents = [Document(page_content=fragment) for fragment in fragments]
            fragment_embeddings = {}
            
            for i, doc in enumerate(documents):
                embedding = self.embedding_model.embed_query(doc.page_content)
                fragment_embeddings[doc.page_content] = embedding
            
            print(f"Созданы эмбеддинги для {len(fragment_embeddings)} фрагментов")
            return fragment_embeddings
        except Exception as e:
            print(f"Ошибка при создании эмбеддингов: {e}")
            return {}
    
    def _get_user_history(self, user_id: str) -> List[Any]:
        
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = [self.system_message]
        return self.conversation_history[user_id]
    
    def _add_to_history(self, user_id: str, message) -> None:
        
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = [self.system_message]
        
        self.conversation_history[user_id].append(message)
        
    
        if len(self.conversation_history[user_id]) > 10:
            self.conversation_history[user_id] = [self.system_message] + self.conversation_history[user_id][-9:]
    
    def find_relevant_fragments(self, query: str, top_n: int = 3) -> List[str]:
        
        try:
            query_embedding = self.embedding_model.embed_query(query)
            
            similarities = {}
            for fragment, fragment_embedding in self.embeddings.items():
                dot_product = sum(q * f for q, f in zip(query_embedding, fragment_embedding))
                query_norm = sum(q ** 2 for q in query_embedding) ** 0.5
                fragment_norm = sum(f ** 2 for f in fragment_embedding) ** 0.5
                similarity = dot_product / (query_norm * fragment_norm) if query_norm * fragment_norm > 0 else 0
                similarities[fragment] = similarity
            
            sorted_fragments = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
            return [fragment for fragment, _ in sorted_fragments]
        except Exception as e:
            print(f"Ошибка при поиске релевантных фрагментов: {e}")
            return []
    
    def process_query(self, user_id: str, query: str) -> str:
        
        try:
            
            user_message = HumanMessage(content=query)
            self._add_to_history(user_id, user_message)
            
           
            conversation_history = self._get_user_history(user_id)
            
            
            relevant_fragments = self.find_relevant_fragments(query)
            
            if not relevant_fragments:
               
                response = self.chat_model.invoke(conversation_history)
                self._add_to_history(user_id, AIMessage(content=response.content))
                return response.content
            
         
            context = "\n\n".join(relevant_fragments)
            context_message = HumanMessage(
                content=f"Вот контекст, который может помочь ответить на последний вопрос пользователя:\n\n{context}\n\nПожалуйста, используй этот контекст для ответа на последний вопрос."
            )
            
            
            messages = conversation_history + [context_message]
            response = self.chat_model.invoke(messages)
            
           
            self._add_to_history(user_id, AIMessage(content=response.content))
            
            return response.content
        except Exception as e:
            print(f"Ошибка при обработке запроса: {e}")
            import traceback
            traceback.print_exc()
            return "Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз."

    def reset_history(self, user_id: str) -> None:
       
        if user_id in self.conversation_history:
            self.conversation_history[user_id] = [self.system_message]
            return True
        return False


def run_telegram_bot():
    
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN не найден в переменных окружения")
        return

   
    bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
    rag_chat = RagChatWithMemory()

 
    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        user_id = str(message.from_user.id)
        welcome_text = (
            "👋 Привет! Я чат-бот школы дизайна, который поможет тебе с вопросами о правилах поведения.\n\n"
            "Вот несколько популярных тем:\n"
            "• Расписание занятий\n"
            "• Правила посещения занятий\n"
            "• Дресс-код\n"
            "• Контактная информация\n\n"
            "Просто задай вопрос, и я постараюсь помочь!"
        )
        bot.send_message(message.chat.id, welcome_text)

    
    @bot.message_handler(commands=['help'])
    def send_help(message):
        help_text = (
            "🔍 Я могу ответить на вопросы о правилах поведения в школе дизайна.\n\n"
            "Вот примеры вопросов, которые вы можете задать:\n"
            "• Во сколько начинаются занятия?\n"
            "• Какие правила посещения занятий?\n"
            "• Как связаться с деканатом?\n"
            "• Можно ли опаздывать на занятия?\n\n"
            "Доступные команды:\n"
            "/start - Начать разговор\n"
            "/help - Показать это сообщение\n"
            "/categories - Показать категории правил\n"
            "/reset - Сбросить историю диалога"
        )
        bot.send_message(message.chat.id, help_text)

    
    @bot.message_handler(commands=['categories'])
    def show_categories(message):
        categories_text = (
            "📚 Основные категории правил:\n\n"
            "• Общие правила\n"
            "• Расписание занятий\n"
            "• Посещаемость\n"
            "• Правила поведения\n"
            "• Дресс-код\n"
            "• Правила использования оборудования\n"
            "• Контактная информация\n\n"
            "Задайте вопрос по интересующей вас категории!"
        )
        bot.send_message(message.chat.id, categories_text)

    
    @bot.message_handler(commands=['reset'])
    def reset_conversation(message):
        user_id = str(message.from_user.id)
        if rag_chat.reset_history(user_id):
            bot.send_message(message.chat.id, "🔄 История диалога сброшена. Начнем заново!")
        else:
            bot.send_message(message.chat.id, "История диалога уже пуста. Можете начать новый разговор!")

    
    @bot.message_handler(func=lambda message: True)
    def handle_message(message):
        user_id = str(message.from_user.id)
        query = message.text
        
        # Индикатор "печатает..."
        bot.send_chat_action(message.chat.id, 'typing')
        
        try:
            answer = rag_chat.process_query(user_id, query)
            bot.send_message(message.chat.id, answer)
        except Exception as e:
            logger.error(f"Ошибка при обработке сообщения: {e}")
            bot.send_message(
                message.chat.id, 
                "Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз."
            )

   
    print("Telegram-бот запущен и готов отвечать на вопросы!")
    bot.infinity_polling()


def main():
    
    run_telegram_bot()

if __name__ == "__main__":
    main() 
