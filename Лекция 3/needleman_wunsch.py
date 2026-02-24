import numpy as np

def needleman_wunsch(seq1, seq2, match_score=1, mismatch_score=-1, gap_penalty=-1):
    """
    Реализация алгоритма Нидлмана-Вунша для глобального выравнивания.
    
    Parameters:
    seq1, seq2: строки с последовательностями
    match_score: очки за совпадение
    mismatch_score: очки за несовпадение
    gap_penalty: штраф за гэп (обычно отрицательный)
    
    Returns:
    aligned_seq1, aligned_seq2: выровненные последовательности
    score: итоговый скор выравнивания
    matrix: матрица скоров (для визуализации)
    """
    
    # Размеры матрицы
    n = len(seq1) + 1
    m = len(seq2) + 1
    
    # Создаем матрицу для скоров
    score_matrix = np.zeros((n, m), dtype=int)
    
    # Создаем матрицу для хранения направления (для трейсинга)
    # 0 - диагональ, 1 - сверху, 2 - слева
    trace_matrix = np.zeros((n, m), dtype=int)
    
    # Инициализация первой строки и первого столбца
    for i in range(1, n):
        score_matrix[i][0] = score_matrix[i-1][0] + gap_penalty
        trace_matrix[i][0] = 1  # сверху
    
    for j in range(1, m):
        score_matrix[0][j] = score_matrix[0][j-1] + gap_penalty
        trace_matrix[0][j] = 2  # слева
    
    # Заполнение матрицы
    for i in range(1, n):
        for j in range(1, m):
            # Считаем три возможных варианта
            
            # Диагональ (match или mismatch)
            if seq1[i-1] == seq2[j-1]:
                diagonal = score_matrix[i-1][j-1] + match_score
            else:
                diagonal = score_matrix[i-1][j-1] + mismatch_score
            
            # Сверху (гэп в seq2)
            up = score_matrix[i-1][j] + gap_penalty
            
            # Слева (гэп в seq1)
            left = score_matrix[i][j-1] + gap_penalty
            
            # Выбираем максимальный
            max_score = max(diagonal, up, left)
            score_matrix[i][j] = max_score
            
            # Запоминаем направление
            if max_score == diagonal:
                trace_matrix[i][j] = 0  # диагональ
            elif max_score == up:
                trace_matrix[i][j] = 1  # сверху
            else:
                trace_matrix[i][j] = 2  # слева
    
    # Трейсинг (обратный ход) для построения выравнивания
    aligned_seq1 = []
    aligned_seq2 = []
    i, j = n-1, m-1
    
    while i > 0 or j > 0:
        if trace_matrix[i][j] == 0:  # диагональ
            aligned_seq1.append(seq1[i-1])
            aligned_seq2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif trace_matrix[i][j] == 1:  # сверху (гэп в seq2)
            aligned_seq1.append(seq1[i-1])
            aligned_seq2.append('-')
            i -= 1
        else:  # слева (гэп в seq1)
            aligned_seq1.append('-')
            aligned_seq2.append(seq2[j-1])
            j -= 1
    
    # Разворачиваем, так как шли с конца
    aligned_seq1 = ''.join(reversed(aligned_seq1))
    aligned_seq2 = ''.join(reversed(aligned_seq2))
    
    return aligned_seq1, aligned_seq2, score_matrix[n-1][m-1], score_matrix

# Функция для красивого вывода матрицы
def print_matrix(seq1, seq2, matrix):
    """Красивый вывод матрицы скоров"""
    print("Матрица скоров:")
    print("    ", " ".join(f"{b:4}" for b in " " + seq2))
    for i, row in enumerate(matrix):
        if i == 0:
            print(f"  {row[0]:4}", " ".join(f"{x:4}" for x in row[1:]))
        else:
            print(f"{seq1[i-1]} {row[0]:4}", " ".join(f"{x:4}" for x in row[1:]))

# Функция для демонстрации примера из лекции
def demonstrate_example():
    """Демонстрация на примере из лекции"""
    print("=" * 60)
    print("АЛГОРИТМ НИДЛМАНА-ВУНША")
    print("=" * 60)
    
    # Пример из лекции
    seq1 = "ATC"
    seq2 = "ACG"
    
    print(f"\nПоследовательность 1: {seq1}")
    print(f"Последовательность 2: {seq2}")
    print(f"Параметры: match=+1, mismatch=-1, gap=-1")
    
    aligned1, aligned2, score, matrix = needleman_wunsch(seq1, seq2)
    
    print("\n" + "=" * 40)
    print("РЕЗУЛЬТАТ ВЫРАВНИВАНИЯ:")
    print("=" * 40)
    print(f"Выравнивание:")
    print(f"1: {aligned1}")
    print(f"2: {aligned2}")
    print(f"\nИтоговый скор: {score}")
    
    print("\n" + "=" * 40)
    print_matrix(seq1, seq2, matrix)
    
    # Визуализация соответствий
    print("\n" + "=" * 40)
    print("ВИЗУАЛИЗАЦИЯ СООТВЕТСТВИЙ:")
    print("=" * 40)
    match_line = ""
    for a, b in zip(aligned1, aligned2):
        if a == b:
            match_line += "|"
        elif a == '-' or b == '-':
            match_line += " "
        else:
            match_line += "."
    print(f"1: {aligned1}")
    print(f"   {match_line}")
    print(f"2: {aligned2}")

# Функция для сравнения разных штрафов
def compare_penalties():
    """Сравнение результатов с разными штрафами"""
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ РАЗНЫХ ШТРАФОВ")
    print("=" * 60)
    
    seq1 = "ATC"
    seq2 = "ACG"
    
    # Разные комбинации штрафов
    penalties = [
        (1, -1, -1, "Стандартные"),
        (2, -2, -1, "Высокий бонус за совпадение"),
        (1, -1, -2, "Высокий штраф за гэп"),
        (1, -3, -1, "Высокий штраф за несовпадение")
    ]
    
    for match, mismatch, gap, name in penalties:
        aligned1, aligned2, score, _ = needleman_wunsch(seq1, seq2, match, mismatch, gap)
        print(f"\n{name} (match={match}, mismatch={mismatch}, gap={gap}):")
        print(f"  {aligned1}")
        print(f"  {aligned2}")
        print(f"  Скор: {score}")

# Функция для демонстрации проблемы глобального выравнивания
def demonstrate_global_alignment_issue():
    """Демонстрация проблемы глобального выравнивания с длинными хвостами"""
    print("\n" + "=" * 60)
    print("ПРОБЛЕМА ГЛОБАЛЬНОГО ВЫРАВНИВАНИЯ")
    print("=" * 60)
    
    # Последовательности с общим участком, но разными концами
    seq1 = "ATCGCGTAGC"  # есть общий участок
    seq2 = "CGCG"        # короткая, но похожа на участок из seq1
    
    print(f"\nПоследовательность 1 (длинная): {seq1}")
    print(f"Последовательность 2 (короткая): {seq2}")
    print("\nПроблема: глобальное выравнивание попытается натянуть короткую")
    print("последовательность на длинную с помощью множества гэпов,")
    print("хотя биологически правильнее было бы выровнять только похожий участок.")
    
    aligned1, aligned2, score, _ = needleman_wunsch(seq1, seq2)
    
    print("\n" + "-" * 40)
    print("РЕЗУЛЬТАТ ГЛОБАЛЬНОГО ВЫРАВНИВАНИЯ:")
    print("-" * 40)
    print(f"1: {aligned1}")
    print(f"2: {aligned2}")
    print(f"Скор: {score}")
    
    # Показываем, где находится реальный похожий участок
    print("\n" + "-" * 40)
    print("РЕАЛЬНЫЙ ПОХОЖИЙ УЧАСТОК (CGCG):")
    print("-" * 40)
    print(f"1: {seq1[3:7]}")
    print(f"2: {seq2}")

if __name__ == "__main__":
    # Демонстрируем основной пример
    demonstrate_example()
    
    # Сравниваем разные штрафы
    compare_penalties()
    
    # Показываем проблему глобального выравнивания
    demonstrate_global_alignment_issue()
    
    # Дополнительный пример из лекции про проблему с инверсией
    print("\n" + "=" * 60)
    print("ПРИМЕР ИЗ ЛЕКЦИИ (ПРОБЛЕМА С ИНВЕРСИЕЙ)")
    print("=" * 60)
    
    # Пример, где есть инверсия (примерно как в лекции)
    seq1 = "ATCGCG"
    seq2 = "GCGATC"
    
    print(f"seq1: {seq1}")
    print(f"seq2: {seq2}")
    print("\nВ seq2 есть участок, комплементарный участку из seq1,")
    print("но глобальное выравнивание не может это учесть")
    
    aligned1, aligned2, score, _ = needleman_wunsch(seq1, seq2)
    print(f"\n1: {aligned1}")
    print(f"2: {aligned2}")
    print(f"Скор: {score}")
    
    print("\n" + "=" * 60)
    print("ВЫВОДЫ:")
    print("=" * 60)
    print("1. Алгоритм Нидлмана-Вунша гарантирует оптимальное глобальное выравнивание")
    print("2. Он всегда находит путь с максимальным скором")
    print("3. Но глобальное выравнивание подходит только для последовательностей,")
    print("   которые похожи по всей длине")
    print("4. Для поиска локальных похожих участков нужен алгоритм Смита-Уотермана")
    print("   (локальное выравнивание)")