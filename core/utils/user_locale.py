# 支持的地区
SUPPORTED_LOCALES = ["en", "de", "it", "zh", "ja", "pt", "fr", "es"]
DEFAULT_LOCALE = "zh"


async def get_user_locale(user_id: str) -> str:
    return DEFAULT_LOCALE


def get_locale_context_prompt(locale: str) -> str:
    """
    生成一段面向本地化的上下文提示词，追加至系统提示词中。

    参数:
        locale: 用户的首选地区 ('en', 'de', 'it', 'zh', 'ja', 'pt', 'fr', 'es')

    返回:
        格式化后的提示词字符串，包含地区指令
    """
    locale_instructions = {
        "en": """## LANGUAGE PREFERENCE
The user has set their preferred language to English. You should respond in English using an informal, semi-personal, and neutral tone. Use casual but professional language throughout your responses.""",
        "de": """## SPRACHPREFERENZ
Der Benutzer hat Deutsch als bevorzugte Sprache eingestellt. Du solltest auf Deutsch antworten und dabei eine informelle, halbpersönliche und neutrale Tonart verwenden. Verwende "du" statt "Sie" und eine lockere aber professionelle Sprache in allen deinen Antworten, Erklärungen und Interaktionen.""",
        "it": """## PREFERENZA LINGUISTICA
L'utente ha impostato l'italiano come lingua preferita. Dovresti rispondere in italiano usando un tono informale, semi-personale e neutro. Usa "tu" invece di "Lei" e un linguaggio casuale ma professionale in tutte le tue risposte, spiegazioni e interazioni.""",
        "zh": """## 语言偏好
用户已将首选语言设置为中文。你应该用中文回复，使用非正式、半个人化且中性的语气。在所有回复、解释和交互中使用随意但专业的语言。""",
        "ja": """## 言語設定
ユーザーは日本語を優先言語に設定しています。日本語で応答し、カジュアルで半個人的かつ中立的なトーンを使用してください。すべての応答、説明、インタラクションでカジュアルだがプロフェッショナルな言語を使用してください。""",
        "pt": """## PREFERÊNCIA DE IDIOMA
O usuário definiu o português como idioma preferido. Você deve responder em português usando um tom informal, semi-pessoal e neutro. Use linguagem casual mas profissional em todas as suas respostas, explicações e interações.""",
        "fr": """## PRÉFÉRENCE DE LANGUE
L'utilisateur a défini le français comme langue préférée. Tu dois répondre en français en utilisant un ton informel, semi-personnel et neutre. Utilise "tu" au lieu de "vous" et un langage décontracté mais professionnel dans toutes tes réponses, explications et interactions.""",
        "es": """## PREFERENCIA DE IDIOMA
El usuario ha establecido el español como idioma preferido. Debes responder en español usando un tono informal, semi-personal y neutro. Usa "tú" en lugar de "usted" y un lenguaje casual pero profesional en todas tus respuestas, explicaciones e interacciones.""",
    }

    return locale_instructions.get(locale, locale_instructions[DEFAULT_LOCALE])
