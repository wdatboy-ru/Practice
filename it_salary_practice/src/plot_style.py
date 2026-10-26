"""Единый стиль графиков и русские подписи категорий."""
import locale
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FIG = Path(__file__).resolve().parents[1] / "reports" / "figures"

# категориальная палитра в фиксированном порядке (проверена на различимость
# при нарушениях цветового зрения), основной цвет — первый
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
# расходящаяся шкала: синий — серый (ноль) — красный
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#184f95", "#86b6ef", "#f0efec", "#f0a3a2", "#b3302f"])

COUNTRY_RU = {
    "United States of America": "США", "Germany": "Германия",
    "United Kingdom of Great Britain and Northern Ireland": "Великобритания",
    "India": "Индия", "France": "Франция", "Canada": "Канада", "Ukraine": "Украина",
    "Netherlands": "Нидерланды", "Brazil": "Бразилия", "Spain": "Испания",
    "Australia": "Австралия", "Italy": "Италия", "Sweden": "Швеция", "Poland": "Польша",
    "Switzerland": "Швейцария", "Austria": "Австрия", "Czech Republic": "Чехия",
    "Denmark": "Дания", "Portugal": "Португалия", "Norway": "Норвегия",
    "Finland": "Финляндия", "Israel": "Израиль", "Belgium": "Бельгия",
    "South Africa": "ЮАР", "Turkey": "Турция", "Russian Federation": "Россия",
    "Romania": "Румыния", "Pakistan": "Пакистан", "New Zealand": "Новая Зеландия",
    "Mexico": "Мексика", "Argentina": "Аргентина", "Japan": "Япония",
    "Ireland": "Ирландия", "Greece": "Греция", "Hungary": "Венгрия", "Bulgaria": "Болгария",
    "China": "Китай", "Other": "Прочие страны",
}
FEATURE_RU = {
    "work_exp": "Стаж работы", "years_code": "Стаж программирования", "age": "Возраст",
    "edu_level": "Образование", "org_size_log": "lg(размер компании)",
    "remote": "Удалённость", "manager": "Руководитель", "n_languages": "Число языков",
    "lang_python": "Python", "lang_sql": "SQL", "lang_javascript": "JavaScript",
    "lang_typescript": "TypeScript", "lang_java": "Java", "lang_csharp": "C#",
    "lang_go": "Go", "lang_rust": "Rust", "lang_cpp": "C++", "lang_php": "PHP",
    "lang_kotlin": "Kotlin", "lang_bash": "Bash/Shell", "ln_salary": "ln(доход)",
}
REMOTE_RU = {
    "In-person": "Только офис",
    "Hybrid (some remote, leans heavy to in-person)": "Гибрид, больше офиса",
    "Hybrid (some in-person, leans heavy to flexibility)": "Гибрид, больше удалёнки",
    "Your choice (very flexible, you can come in when you want or just as needed)": "Свободный выбор",
    "Remote": "Полностью удалённо",
}
EDU_RU = {
    "Primary/elementary school": "Начальное",
    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": "Среднее (школа)",
    "Some college/university study without earning a degree": "Неоконченное высшее",
    "Associate degree (A.A., A.S., etc.)": "Associate degree",
    "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": "Бакалавр",
    "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": "Магистр",
    "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": "PhD и аналоги",
    "Other (please specify):": "Другое",
}
ORG_RU = {
    "Just me - I am a freelancer, sole proprietor, etc.": "1 (сам)",
    "Less than 20 employees": "<20", "20 to 99 employees": "20–99",
    "100 to 499 employees": "100–499", "500 to 999 employees": "500–999",
    "1,000 to 4,999 employees": "1 000–4 999", "5,000 to 9,999 employees": "5 000–9 999",
    "10,000 or more employees": "10 000+", "I don’t know": "Не знаю",
}
DEV_RU = {
    "AI/ML engineer": "ML-инженер", "Academic researcher": "Научный сотрудник",
    "Applied scientist": "Прикладной исследователь",
    "Architect, software or solutions": "Архитектор ПО",
    "Cloud infrastructure engineer": "Инженер облачной инфраструктуры",
    "Cybersecurity or InfoSec professional": "Специалист по ИБ",
    "Data engineer": "Инженер данных", "Data or business analyst": "Аналитик данных/бизнеса",
    "Data scientist": "Data scientist", "Database administrator or engineer": "Администратор БД",
    "DevOps engineer or professional": "DevOps-инженер",
    "Developer, AI apps or physical AI": "Разработчик ИИ-приложений",
    "Developer, QA or test": "Тестировщик (QA)", "Developer, back-end": "Back-end разработчик",
    "Developer, desktop or enterprise applications": "Разработчик desktop/корп. ПО",
    "Developer, embedded applications or devices": "Разработчик встраиваемых систем",
    "Developer, front-end": "Front-end разработчик", "Developer, full-stack": "Full-stack разработчик",
    "Developer, game or graphics": "Разработчик игр и графики", "Developer, mobile": "Мобильный разработчик",
    "Engineering manager": "Руководитель разработки",
    "Financial analyst or engineer": "Финансовый аналитик/инженер",
    "Founder, technology or otherwise": "Основатель компании", "Other (please specify):": "Другое",
    "Product manager": "Продакт-менеджер", "Project manager": "Руководитель проектов",
    "Retired": "На пенсии", "Senior executive (C-suite, VP, etc.)": "Топ-менеджер (C-level, VP)",
    "Student": "Студент", "Support engineer or analyst": "Инженер поддержки",
    "System administrator": "Системный администратор",
    "UX, Research Ops or UI design professional": "UX/UI-специалист",
}


def setup():
    # десятичная запятая на осях
    try:
        locale.setlocale(locale.LC_NUMERIC, "ru_RU.UTF-8")
        plt.rcParams["axes.formatter.use_locale"] = True
    except locale.Error:
        pass
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.prop_cycle": plt.cycler(color=SERIES),
        "lines.linewidth": 2, "figure.dpi": 100, "savefig.dpi": 200,
        "savefig.bbox": "tight", "legend.frameon": False,
    })
    FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", facecolor="white")
    plt.close(fig)


def fmt_int(n):
    """Целое с пробелом между разрядами: 16891 -> «16 891»."""
    return f"{int(n):,}".replace(",", " ")


def fmt_dec(x, digits=2):
    """Дробное по-русски: запятая и знак минуса."""
    x = round(x, digits) + 0.0  # убирает «минус ноль»
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def usd_k(value):
    """Подпись суммы в тысячах долларов: 81 210 -> «81 тыс.»."""
    return f"{fmt_int(round(value / 1000))} тыс."
