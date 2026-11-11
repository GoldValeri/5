"""
Лабораторная работа № 5. Вариант 6.

Хеширование и экспериментальное исследование методов разрешения коллизий.

Параметры варианта 6 (табл. 1 методички):
  - тип ключей: строки (артикулы вида ABC-12345)
  - хеш-функция: П — полиномиальная для строк
  - разрешение коллизий: Ц (метод цепочек) и Дв (двойное хеширование)
  - n = 30 000 элементов, m = 49 999 (простое)
  - доп. задание В6: исследование кластеризации при линейном пробировании

Дисциплина: Основы алгоритмизации и программирование
Студент: Голдаева Валерия Валентиновна, группа БИ 2-1
"""

# ---------------------------------------------------------------------------
# Стандартные библиотеки
# ---------------------------------------------------------------------------
import random          # генерация артикулов и запросов
import string          # буквы для префикса артикула
import time            # time.perf_counter — точный таймер
from typing import Any, Callable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Константы варианта
# ---------------------------------------------------------------------------
M = 49_999   # размер таблицы (простое число — важно для метода деления/двойного хеша)
N = 30_000   # число вставляемых элементов
P_BASE = 31  # основание полинома для строкового хеша (простое)


# ===========================================================================
# 1. ХЕШ-ФУНКЦИИ
# ===========================================================================

def poly_hash(key: str, m: int = M, p: int = P_BASE) -> int:
    """
    Полиномиальное хеширование строки (вариант: П).

    h(s) = (s[0]*p^0 + s[1]*p^1 + ... + s[L-1]*p^{L-1}) mod m

    Зависит от ВСЕХ символов ключа → равномерное распределение.
    """
    h = 0          # накопленная сумма
    pw = 1         # текущая степень p^i
    for ch in key:
        # ord(ch) — код символа; прибавляем вклад i-го символа
        h = (h + ord(ch) * pw) % m
        # переходим к следующей степени: p^{i+1} = p^i * p
        pw = (pw * p) % m
    return h


def bad_hash(key: str, m: int = M) -> int:
    """
    Заведомо плохая хеш-функция: учитывает только первый символ.

    Даёт всего ~26 различных значений для латинских префиксов →
    огромные коллизии. Используется для сравнения с poly_hash.
    """
    if not key:
        return 0
    return ord(key[0]) % m


# ===========================================================================
# 2. ГЕНЕРАЦИЯ ДАННЫХ
# ===========================================================================

def make_articles(n: int, seed: int = 42) -> List[str]:
    """
    Генерирует n уникальных артикулов вида «ABC-12345».

    Префикс — 3 заглавные латинские буквы, суффикс — 5 цифр.
    """
    rng = random.Random(seed)   # фиксированный seed → воспроизводимость
    arts = set()                # множество гарантирует уникальность
    while len(arts) < n:
        # rng.choices — случайные буквы с возвращением
        prefix = ''.join(rng.choices(string.ascii_uppercase, k=3))
        num = rng.randint(10000, 99999)
        arts.add(f'{prefix}-{num}')
    return list(arts)


# ===========================================================================
# 3. МЕТОД ЦЕПОЧЕК
# ===========================================================================

class ChainHashTable:
    """
    Хеш-таблица с разрешением коллизий методом цепочек.

    Каждая ячейка slots[i] — список пар (ключ, значение).
    Коэффициент заполнения α = n/m может быть > 1.
    Средняя трудоёмкость поиска: Θ(1 + α).
    """

    def __init__(self, m: int = M, hash_fn: Callable = poly_hash):
        self.m = m
        # создаём m пустых списков (цепочек)
        self.slots: List[List[Tuple[str, Any]]] = [[] for _ in range(m)]
        self.n = 0              # число хранимых элементов
        self.probes = 0         # счётчик просмотренных элементов (для измерений)
        self.hash_fn = hash_fn  # используемая хеш-функция

    def insert(self, key: str, value: Any) -> None:
        """Вставка / обновление пары (key, value)."""
        # вычисляем индекс ячейки
        idx = self.hash_fn(key, self.m)
        chain = self.slots[idx]
        # ищем ключ в цепочке (на случай обновления)
        for i, (k, _) in enumerate(chain):
            self.probes += 1          # одно сравнение ключей
            if k == key:
                chain[i] = (key, value)  # ключ уже есть — обновляем значение
                return
        # ключа нет — добавляем в конец цепочки
        chain.append((key, value))
        self.n += 1

    def search(self, key: str) -> Optional[Any]:
        """Поиск значения по ключу. None, если не найден."""
        idx = self.hash_fn(key, self.m)
        for k, v in self.slots[idx]:
            self.probes += 1
            if k == key:
                return v
        return None

    def chain_lengths(self) -> List[int]:
        """Длины всех цепочек — для анализа распределения."""
        return [len(c) for c in self.slots]


# ===========================================================================
# 4. ОТКРЫТАЯ АДРЕСАЦИЯ: ДВОЙНОЕ ХЕШИРОВАНИЕ
# ===========================================================================

class DoubleHashTable:
    """
    Хеш-таблица с открытой адресацией и двойным хешированием.

    Последовательность проб:
        pos_i = (h1(k) + i * h2(k)) mod m

    h2 выбирается так, чтобы шаг был взаимно прост с m
    (m простое → достаточно h2 ∈ [1; m-1]).

    Обязательно α < 1. Среднее число проб при неуспехе ≈ 1/(1-α).
    """

    # маркеры пустой и удалённой ячейки (tombstone)
    EMPTY = object()
    DELETED = object()

    def __init__(self, m: int = M, hash_fn: Callable = poly_hash):
        self.m = m
        self.hash_fn = hash_fn
        # массив ключей, изначально все EMPTY
        self.keys: List[Any] = [self.EMPTY] * m
        self.values: List[Any] = [None] * m
        self.n = 0
        self.probes = 0

    def _h1(self, key: str) -> int:
        """Основная хеш-функция — начальная позиция."""
        return self.hash_fn(key, self.m)

    def _h2(self, key: str) -> int:
        """
        Вторая хеш-функция — шаг пробирования.
        Берём хеш от перевёрнутой строки по модулю (m-1) и прибавляем 1,
        чтобы шаг ∈ [1; m-1] и был ненулевым.
        """
        return 1 + self.hash_fn(key[::-1], self.m - 1)

    def insert(self, key: str, value: Any) -> None:
        """Вставка с двойным хешированием."""
        pos = self._h1(key)
        step = self._h2(key)
        for _ in range(self.m):       # не более m попыток
            self.probes += 1
            k = self.keys[pos]
            # свободная или ранее удалённая ячейка — можно писать
            if k is self.EMPTY or k is self.DELETED:
                self.keys[pos] = key
                self.values[pos] = value
                self.n += 1
                return
            # ключ уже есть — обновляем значение
            if k == key:
                self.values[pos] = value
                return
            # коллизия — шаг двойного хеширования
            pos = (pos + step) % self.m
        raise RuntimeError('Таблица переполнена')

    def search(self, key: str) -> Optional[Any]:
        """Поиск по той же последовательности проб."""
        pos = self._h1(key)
        step = self._h2(key)
        for _ in range(self.m):
            self.probes += 1
            k = self.keys[pos]
            # EMPTY означает конец цепочки проб — ключа нет
            if k is self.EMPTY:
                return None
            # DELETED пропускаем (tombstone), продолжаем пробы
            if k is not self.DELETED and k == key:
                return self.values[pos]
            pos = (pos + step) % self.m
        return None


# ===========================================================================
# 5. ЛИНЕЙНОЕ ПРОБИРОВАНИЕ (для доп. задания В6 — кластеризация)
# ===========================================================================

class LinearProbeTable:
    """
    Открытая адресация с линейным пробированием: pos = (h(k) + i) mod m.

    Используется только для исследования первичной кластеризации (В6).
    """

    EMPTY = object()

    def __init__(self, m: int):
        self.m = m
        self.keys: List[Any] = [self.EMPTY] * m
        self.n = 0

    def insert(self, key: str) -> None:
        pos = poly_hash(key, self.m)
        # линейный шаг +1, пока не найдём свободную ячейку
        while self.keys[pos] is not self.EMPTY:
            if self.keys[pos] == key:
                return
            pos = (pos + 1) % self.m
        self.keys[pos] = key
        self.n += 1

    def cluster_lengths(self) -> List[int]:
        """
        Длины непрерывных занятых участков (кластеров).

        Учитываем «кольцевой» кластер, соединяющий конец и начало массива.
        """
        occupied = [k is not self.EMPTY for k in self.keys]
        lengths: List[int] = []
        current = 0
        for flag in occupied + [False]:   # False в конце сбрасывает хвост
            if flag:
                current += 1
            elif current:
                lengths.append(current)
                current = 0
        # склейка кольцевого кластера
        if occupied[0] and occupied[-1] and len(lengths) >= 2:
            lengths[0] += lengths[-1]
            lengths.pop()
        return lengths


# ===========================================================================
# 6. ИЗМЕРЕНИЯ
# ===========================================================================

def benchmark(table, keys: Sequence[str], queries: Sequence[str]):
    """
    Возвращает (время вставки, время поиска, среднее число проб на запрос).

    Перед поиском обнуляем table.probes, чтобы считать только поисковые пробы.
    """
    t0 = time.perf_counter()
    for k in keys:
        table.insert(k, k)
    t_insert = time.perf_counter() - t0

    table.probes = 0
    t0 = time.perf_counter()
    for q in queries:
        table.search(q)
    t_search = time.perf_counter() - t0

    avg_probes = table.probes / len(queries)
    return t_insert, t_search, avg_probes


# ===========================================================================
# 7. ОСНОВНЫЕ ЭКСПЕРИМЕНТЫ
# ===========================================================================

def run_main() -> None:
    """Основной эксперимент: цепочки и двойное хеширование, сравнение с dict."""
    print('=' * 70)
    print(f'Вариант 6: строки (артикулы), poly-hash, Ц + Дв')
    print(f'n = {N}, m = {M}, α = {N / M:.3f}')
    print('=' * 70)

    keys = make_articles(N)
    queries = random.Random(1).choices(keys, k=10_000)

    for name, table in [
        ('Метод цепочек', ChainHashTable()),
        ('Двойное хеширование', DoubleHashTable()),
    ]:
        ti, ts, probes = benchmark(table, keys, queries)
        print(f'{name:<22} вставка {ti:.3f} с, поиск {ts:.3f} с, '
              f'проб/запрос {probes:.2f}')

    # эталон — встроенный dict
    t0 = time.perf_counter()
    d = {k: k for k in keys}
    ti = time.perf_counter() - t0
    t0 = time.perf_counter()
    for q in queries:
        d.get(q)
    ts = time.perf_counter() - t0
    print(f'{"dict (эталон)":<22} вставка {ti:.3f} с, поиск {ts:.3f} с')

    # статистика цепочек
    ch = ChainHashTable()
    for k in keys:
        ch.insert(k, k)
    lengths = ch.chain_lengths()
    print(f'\nЦепочки: max = {max(lengths)}, пустых = {lengths.count(0)} '
          f'из {M}, средняя длина = {sum(lengths) / len(lengths):.3f}')


def run_good_vs_bad() -> None:
    """Сравнение качественной (poly) и плохой (первый символ) хеш-функций."""
    print('\n' + '=' * 70)
    print('Качественная vs плохая хеш-функция')
    print('=' * 70)
    keys = make_articles(N)
    for fn, name in [(poly_hash, 'poly_hash'), (bad_hash, 'bad_hash (1-й символ)')]:
        slots = [0] * M
        for k in keys:
            slots[fn(k)] += 1
        nonempty = sum(1 for x in slots if x > 0)
        print(f'{name:<28} ненулевых ячеек = {nonempty}, max в ячейке = {max(slots)}')


def run_b6_clustering() -> None:
    """
    Доп. задание В6: кластеризация при линейном пробировании.

    Заполняем таблицу до разных α и смотрим распределение длин кластеров.
    """
    print('\n' + '=' * 70)
    print('В6: кластеризация при линейном пробировании')
    print('=' * 70)
    keys = make_articles(N)
    # для α=0.9 нужно больше ключей
    extra = make_articles(M, seed=99)

    print(f'{"α":>6}  {"кластеров":>10}  {"max":>6}  {"средняя":>8}')
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        n_fill = int(alpha * M)
        pool = (keys + extra)[:n_fill]
        lp = LinearProbeTable(M)
        for k in pool:
            lp.insert(k)
        cl = lp.cluster_lengths()
        mean = sum(cl) / len(cl) if cl else 0.0
        print(f'{alpha:6.1f}  {len(cl):10d}  {max(cl) if cl else 0:6d}  {mean:8.2f}')


def self_test() -> None:
    """Проверка корректности на маленьком наборе."""
    keys = ['AAA-11111', 'BBB-22222', 'AAA-11111']  # повтор для update
    ch = ChainHashTable(m=97)
    dh = DoubleHashTable(m=97)
    for k in keys:
        ch.insert(k, k)
        dh.insert(k, k)
    assert ch.search('AAA-11111') == 'AAA-11111'
    assert ch.search('ZZZ-00000') is None
    assert dh.search('BBB-22222') == 'BBB-22222'
    assert dh.search('ZZZ-00000') is None
    print('self_test: OK')


if __name__ == '__main__':
    self_test()
    run_main()
    run_good_vs_bad()
    run_b6_clustering()
