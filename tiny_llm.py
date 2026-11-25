"""
Простейшая символьная языковая модель с нуля.

Шаги (как в file.txt):
  1. Подготовка текста
  2. Перевод в числа (токенизация по символам)
  3. Создание примеров для обучения
  4. Выбор архитектуры: bigram (из примера) или mini-GPT (с механизмом внимания)
  5. Обучение
  6. Генерация текста

Запуск:
  python tiny_llm.py                          # bigram-модель на «Войне и мире», быстро
  python tiny_llm.py --model gpt              # мини-GPT, дольше, но заметно лучше
  python tiny_llm.py --model gpt --steps 5000
  python tiny_llm.py --model gpt --generate-only --prompt "Князь Андрей"
  python tiny_llm.py --data "C:\\путь\\к\\файлу.txt" --model gpt --prompt "Жил-был"
  python tiny_llm.py --data книга1.txt книга2.txt --name мой-микс --model gpt
  python tiny_llm.py --data "C:\\Тексты" --model gpt   # все .txt из папки
  python tiny_llm.py --name tiny_llm_two_filesV1_gpt.pt --generate-only --prompt "Таня"

Подробнее: python tiny_llm.py --help
"""

import argparse
import sys
from pathlib import Path

# Чтобы русский текст не ломался при выводе в файл (python tiny_llm.py > out.txt)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent

# --------------------------------------------------------------------------
# Параметры и справка (--help)
# Разбираем аргументы ДО импорта torch, чтобы --help показывался мгновенно.
# --------------------------------------------------------------------------
DESCRIPTION = """\
Tiny LLM — простейшая символьная языковая модель, обученная с нуля.

Модель читает один или несколько текстовых файлов и учится по предыдущим
символам угадывать следующий. Потом она дописывает текст «в стиле» этих
текстов, начиная с вашей фразы.

Как работает скрипт:
  1. Подготовка текста — читает файлы (UTF-8 или Windows-1251), убирает
     лишние пробелы и редкие символы, склеивает тексты через пустую строку.
  2. Перевод в числа  — каждому символу присваивается номер.
  3. Примеры          — режет текст на случайные кусочки; 90% каждого
                        файла идёт в обучение, последние 10% — в проверку
                        (так проверка честная для всех файлов сразу).
  4. Архитектура      — bigram: смотрит только на 1 предыдущий символ
                        (быстро, но выходит «тарабарщина»);
                        gpt: мини-трансформер с механизмом внимания, видит
                        128 предыдущих символов (дольше, но гораздо лучше).
  5. Обучение         — каждые 250 шагов печатает ошибку (loss) на train и
                        val. Чем меньше, тем лучше. Если train падает, а val
                        растёт — модель заучивает текст (мало данных).
  6. Генерация        — дописывает --length символов после --prompt.

Обученная модель сохраняется рядом со скриптом в файл
tiny_llm_<имя>_<модель>.pt, например tiny_llm_voina-i-mir_gpt.pt.
<имя> — это --name, а если он не задан: имя файла (один файл), имя папки
(одна папка) или имена файлов через «+» (несколько файлов).
Повторное обучение с тем же именем и моделью ПЕРЕЗАПИСЫВАЕТ файл модели.
Вместе с весами сохраняется словарь символов и список файлов, поэтому для
--generate-only сами тексты не нужны — достаточно --name (или того же --data).

Какую модель берёт --generate-only:
  Скрипт НЕ выбирает «последнюю» модель из папки. Он собирает имя файла
  tiny_llm_<имя>_<модель>.pt из параметров и открывает ровно его:
    • <имя>    — из --name, иначе из --data (по умолчанию voina-i-mir);
    • <модель> — из --model (по умолчанию bigram!).
  В --name можно написать и полное имя файла модели
  (tiny_llm_two_filesV1_gpt.pt) — тогда --model тоже берётся из него.
  Если модели нет, скрипт покажет список имеющихся в папке моделей.
"""

EXAMPLES = """\
примеры:
  Обучение:
    py -3.12 tiny_llm.py
        bigram на «Войне и мире» (около 5 секунд)
    py -3.12 tiny_llm.py --model gpt
        мини-GPT, 3000 шагов (около 15 минут на процессоре)
    py -3.12 tiny_llm.py --model gpt --steps 10000
        учить дольше — текст получится связнее
    py -3.12 tiny_llm.py --model gpt --steps 200
        быстрая проверка, что всё работает

  Свой текст:
    py -3.12 tiny_llm.py --data skazki.txt
        файл из папки скрипта
    py -3.12 tiny_llm.py --data "C:\\Тексты\\стихи.txt" --model gpt --prompt "Я помню"
        полный путь (в кавычках, если есть пробелы)

  Несколько текстов в одной модели:
    py -3.12 tiny_llm.py --data voina-i-mir.txt anna-karenina.txt --model gpt
        файлы через пробел; модель: tiny_llm_voina-i-mir+anna-karenina_gpt.pt
    py -3.12 tiny_llm.py --data voina-i-mir.txt anna-karenina.txt --name tolstoy --model gpt
        то же, но с коротким именем модели: tiny_llm_tolstoy_gpt.pt
    py -3.12 tiny_llm.py --data "C:\\Тексты\\Толстой" --model gpt
        все .txt из папки; модель называется по имени папки (Толстой)
    py -3.12 tiny_llm.py --data "C:\\Тексты\\*.txt" --name все-тексты --model gpt
        файлы по маске (* — любые символы, ? — один символ)

  Генерация без обучения (модель уже обучена):
    py -3.12 tiny_llm.py --model gpt --generate-only
    py -3.12 tiny_llm.py --model gpt --generate-only --prompt "Наташа" --length 1000
    py -3.12 tiny_llm.py --data skazki.txt --model gpt --generate-only --prompt "Жил-был"
        для своего текста укажите тот же --data, что и при обучении
    py -3.12 tiny_llm.py --name tolstoy --model gpt --generate-only --prompt "Анна"
        модель из нескольких файлов — достаточно её имени
    py -3.12 tiny_llm.py --name tiny_llm_two_filesV1_gpt.pt --generate-only --prompt "Таня"
        полное имя файла модели; --model указывать не нужно

  Настройка генерации:
    py -3.12 tiny_llm.py --model gpt --generate-only --temperature 0.5
        осторожнее и однообразнее
    py -3.12 tiny_llm.py --model gpt --generate-only --temperature 1.3
        смелее и безумнее

замечания:
  • Нужны только .txt файлы; .docx и .pdf сначала сохраните как текст.
  • Для осмысленного результата нужно от нескольких сотен тысяч символов.
  • Символы в --prompt, которых не было в обучающем тексте, пропускаются.
  • Каждый файл влияет на модель пропорционально своему размеру: большая
    книга «перевесит» маленькую. Доли файлов печатаются перед обучением.
"""

parser = argparse.ArgumentParser(
    prog="tiny_llm.py",
    description=DESCRIPTION,
    epilog=EXAMPLES,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    add_help=False,
)
parser._optionals.title = "параметры"
parser.add_argument("-h", "--help", action="help", help="показать эту справку и выйти")
parser.add_argument("--data", nargs="+", default=[str(HERE / "voina-i-mir.txt")],
                    metavar="ФАЙЛ",
                    help="один или несколько .txt файлов, папок (берутся все .txt внутри) "
                         "или масок вида *.txt (по умолчанию: voina-i-mir.txt)")
parser.add_argument("--name", metavar="ИМЯ",
                    help="имя модели для файла tiny_llm_<ИМЯ>_<модель>.pt "
                         "(по умолчанию: по имени файла, папки или файлов через «+»). "
                         "Можно указать и полное имя файла модели, например "
                         "tiny_llm_two_filesV1_gpt.pt — тогда и --model берётся из него")
parser.add_argument("--model", choices=["bigram", "gpt"], default="bigram",
                    help="архитектура: bigram — простая и быстрая, "
                         "gpt — мини-трансформер (по умолчанию: bigram)")
parser.add_argument("--steps", type=int, default=3000, metavar="N",
                    help="сколько шагов обучения (по умолчанию: 3000)")
parser.add_argument("--prompt", default="Князь Андрей", metavar="ТЕКСТ",
                    help="начало текста для генерации (по умолчанию: «Князь Андрей»)")
parser.add_argument("--length", type=int, default=500, metavar="N",
                    help="сколько символов сгенерировать (по умолчанию: 500)")
parser.add_argument("--temperature", type=float, default=0.8, metavar="T",
                    help="«смелость» генерации: <1 — осторожнее, >1 — фантазийнее "
                         "(по умолчанию: 0.8)")
parser.add_argument("--generate-only", action="store_true",
                    help="не обучать, а загрузить сохранённую модель и сразу генерировать")
args = parser.parse_args()

import re

# Разрешаем писать в --name полное имя файла модели, например
# tiny_llm_two_filesV1_gpt.pt или tiny_llm_two_filesV1_gpt — отрезаем лишнее,
# а тип модели (gpt/bigram) берём из имени файла.
if args.name:
    m = re.fullmatch(r"(?:tiny_llm_)?(.+?)(?:_(bigram|gpt))?(?:\.pt)?", Path(args.name).name)
    if Path(args.name).name.startswith("tiny_llm_") or args.name.endswith(".pt"):
        args.name = m.group(1)
        if m.group(2) and m.group(2) != args.model:
            print(f"Тип модели взят из имени файла: {m.group(2)}")
            args.model = m.group(2)
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

# Гиперпараметры для каждой архитектуры
if args.model == "bigram":
    block_size = 16     # длина кусочка текста
    batch_size = 32     # сколько кусочков за один шаг
    learning_rate = 0.05
    max_steps = args.steps
else:
    block_size = 128    # сколько символов модель «видит» в прошлом
    batch_size = 32
    learning_rate = 1e-3
    max_steps = args.steps
    n_embd = 128        # размер вектора для каждого символа
    n_head = 4          # голов внимания
    n_layer = 4         # блоков трансформера
    dropout = 0.1

eval_every = 250
eval_batches = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)


# --------------------------------------------------------------------------
# Какие файлы читать и как назвать модель
# --------------------------------------------------------------------------
def find_files(items):
    """Превращает список из --data (файлы, папки, маски) в список .txt файлов."""
    files = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            found = sorted(p.glob("*.txt"))
            if not found:
                raise SystemExit(f"В папке {p} нет .txt файлов.")
        elif any(c in p.name for c in "*?["):
            found = sorted(p.parent.glob(p.name))     # PowerShell сам маски не раскрывает
            if not found:
                raise SystemExit(f"По маске {item} ничего не найдено.")
        elif p.is_file():
            found = [p]
        else:
            raise SystemExit(f"Файл не найден: {item}\n"
                             f"Положите текст в .txt рядом со скриптом или укажите путь "
                             f"через --data (подробнее: README.md, раздел «Данные»).")
        for f in found:
            if f.resolve() not in [x.resolve() for x in files]:   # без повторов
                files.append(f)
    return files


files = find_files(args.data)
if args.name:
    run_name = args.name
elif len(args.data) == 1 and Path(args.data[0]).is_dir():
    run_name = Path(args.data[0]).resolve().name
elif len(files) == 1:
    run_name = files[0].stem
else:
    run_name = "+".join(f.stem for f in files)[:80]
CHECKPOINT = HERE / f"tiny_llm_{run_name}_{args.model}.pt"


# --------------------------------------------------------------------------
# 1. Подготовка текста
# --------------------------------------------------------------------------
def read_text(path):
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")        # обычная кодировка
    except UnicodeDecodeError:
        text = raw.decode("cp1251")           # старые русские .txt часто в Windows-1251
    text = text.replace("\xa0", " ")          # неразрывные пробелы -> обычные
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)       # лишние пробелы
    text = re.sub(r"\n{3,}", "\n\n", text)    # лишние пустые строки
    return text.strip()


def load_texts():
    """Читает все файлы, печатает их доли и выбрасывает редкие символы."""
    texts = [read_text(f) for f in files]
    total = sum(len(t) for t in texts)
    print(f"Файлов: {len(files)}")
    for f, t in zip(files, texts):
        print(f"  {f.name:40s} {len(t):>12,} символов  ({100 * len(t) / max(total, 1):5.1f}%)")

    # Выбрасываем редкие символы (реже 1 раза на 30 000 символов всего корпуса):
    # так словарь меньше и чище. Для маленьких текстов порог = 1 (ничего не выкидываем).
    min_count = max(1, total // 30000)
    counts = Counter("".join(texts))
    texts = ["".join(ch for ch in t if counts[ch] >= min_count) for t in texts]

    total = sum(len(t) for t in texts)
    if total < 20 * block_size:
        raise SystemExit(f"Текста слишком мало ({total} символов). "
                         f"Нужно хотя бы {20 * block_size}, а лучше от 100 000.")
    return texts


checkpoint = None
if args.generate_only:
    if not CHECKPOINT.exists():
        available = sorted(p.name for p in HERE.glob("tiny_llm_*.pt"))
        hint = ("\nЕсть модели:\n  " + "\n  ".join(available)) if available else ""
        raise SystemExit(f"Нет обученной модели {CHECKPOINT.name}. "
                         f"Сначала запустите обучение без --generate-only "
                         f"или укажите имя существующей модели в --name.{hint}")
    checkpoint = torch.load(CHECKPOINT, map_location=device)

# --------------------------------------------------------------------------
# 2. Перевод в числа
# --------------------------------------------------------------------------
if checkpoint is not None and "chars" in checkpoint:
    # Словарь берём из сохранённой модели — тексты для генерации не нужны
    chars = checkpoint["chars"]
    texts = None
else:
    texts = load_texts()
    chars = sorted(set("".join(texts)))

vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}   # символ -> номер
itos = {i: ch for i, ch in enumerate(chars)}   # номер -> символ


def encode(s):
    return [stoi[c] for c in s if c in stoi]


def decode(ids):
    return "".join(itos[i] for i in ids)


print(f"Словарь: {vocab_size} символов, устройство: {device}")

# --------------------------------------------------------------------------
# 3. Примеры для обучения.
# Из КАЖДОГО файла 90% идёт в обучение, последние 10% — в проверку.
# Если делить уже склеенный текст, проверка состояла бы только из
# последнего файла и ничего не говорила бы об остальных.
# --------------------------------------------------------------------------
SEP = "\n\n"   # между файлами — пустая строка, как граница «главы»
if texts is not None:
    train_data = torch.tensor(encode(SEP.join(t[:int(0.9 * len(t))] for t in texts)),
                              dtype=torch.long)
    val_data = torch.tensor(encode(SEP.join(t[int(0.9 * len(t)):] for t in texts)),
                            dtype=torch.long)
    print(f"Обучение: {len(train_data):,} символов, проверка: {len(val_data):,} символов")


def get_batch(split):
    """Случайные кусочки текста x и те же кусочки, сдвинутые на 1 символ, y."""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# --------------------------------------------------------------------------
# 4. Архитектуры
# --------------------------------------------------------------------------
class BigramLM(nn.Module):
    """Модель из примера: смотрит только на ОДИН предыдущий символ.
    Вся «память» — таблица vocab_size x vocab_size."""

    def __init__(self):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.table(idx)  # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss


class Head(nn.Module):
    """Одна голова самовнимания: каждый символ «смотрит» на предыдущие
    и решает, какие из них важны для предсказания."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Маска: запрещаем смотреть в будущее
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        w = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # насколько символы «похожи»
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = self.dropout(F.softmax(w, dim=-1))
        return w @ v


class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class Block(nn.Module):
    """Блок трансформера: внимание (общение символов) + MLP (обдумывание)."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHead()
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Маленький GPT: та же идея, что в настоящих LLM, только в миниатюре."""

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)   # что за символ
        self.pos_emb = nn.Embedding(block_size, n_embd)   # на какой он позиции
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0):
    """6. Генерация: предсказываем следующий символ, приклеиваем, повторяем."""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]           # модель видит не больше block_size символов
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature   # берём предсказание для последнего символа
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_batches)
        for k in range(eval_batches):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


model = (BigramLM() if args.model == "bigram" else MiniGPT()).to(device)
print(f"Модель: {args.model}, параметров: {sum(p.numel() for p in model.parameters()):,}")

# --------------------------------------------------------------------------
# 5. Обучение
# --------------------------------------------------------------------------
if args.generate_only:
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        print(f"Загружено из {CHECKPOINT.name}, обучена на: {', '.join(checkpoint['files'])}")
    else:  # старый формат: только веса (словарь построен из --data выше)
        model.load_state_dict(checkpoint)
        print(f"Загружено из {CHECKPOINT.name}")
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    t0 = time.time()
    for step in range(max_steps + 1):
        if step % eval_every == 0 or step == max_steps:
            l = estimate_loss(model)
            print(f"шаг {step:5d} | ошибка train {l['train']:.3f} | val {l['val']:.3f} "
                  f"| {time.time() - t0:.0f} c")
        if step == max_steps:
            break

        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()       # вычисляем, как поправить веса
        optimizer.step()      # обновляем веса

    # Сохраняем веса + словарь + список файлов: этого достаточно для генерации
    torch.save({"model": model.state_dict(),
                "chars": chars,
                "model_type": args.model,
                "files": [f.name for f in files]}, CHECKPOINT)
    print(f"Модель сохранена в {CHECKPOINT.name}")

# --------------------------------------------------------------------------
# 6. Генерация текста
# --------------------------------------------------------------------------
start = torch.tensor([encode(args.prompt) or [stoi.get(" ", 0)]], dtype=torch.long, device=device)
out = generate(model, start, args.length, args.temperature)[0].tolist()
print("\n" + "=" * 60)
print(decode(out))
print("=" * 60)
