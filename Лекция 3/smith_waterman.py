import numpy as np

def smith_waterman(seq1, seq2, match_score=2, mismatch_score=-1, gap_penalty=-2):
    """
    Реализация алгоритма Смита-Уотермана для локального выравнивания.
    
    Parameters:
    seq1, seq2: строки с последовательностями
    match_score: очки за совпадение
    mismatch_score: очки за несовпадение
    gap_penalty: штраф за гэп (отрицательное число)
    
    Returns:
    aligned_seq1, aligned_seq2: выровненные последовательности (только локальный участок)
    max_score: максимальный скор выравнивания
    max_pos: позиция максимума (i, j)
    matrix: матрица скоров
    """
    
    # Размеры матрицы
    n = len(seq1) + 1
    m = len(seq2) + 1
    
    # Создаем матрицу для скоров
    score_matrix = np.zeros((n, m), dtype=int)
    
    # Создаем матрицу для хранения направления (для трейсинга)
    # 0 - диагональ, 1 - сверху, 2 - слева, -1 - старт (ноль)
    trace_matrix = np.ones((n, m), dtype=int) * -1
    
    # Переменные для отслеживания максимума
    max_score = 0
    max_pos = (0, 0)
    
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
            
            # КЛЮЧЕВОЕ ОТЛИЧИЕ: добавляем 0 как вариант
            # Это позволяет начинать новое выравнивание
            max_score_cell = max(0, diagonal, up, left)
            score_matrix[i][j] = max_score_cell
            
            # Запоминаем направление (если не ноль)
            if max_score_cell > 0:
                if max_score_cell == diagonal:
                    trace_matrix[i][j] = 0  # диагональ
                elif max_score_cell == up:
                    trace_matrix[i][j] = 1  # сверху
                elif max_score_cell == left:
                    trace_matrix[i][j] = 2  # слева
            
            # Обновляем глобальный максимум
            if max_score_cell > max_score:
                max_score = max_score_cell
                max_pos = (i, j)
    
    # Трейсинг (обратный ход) от максимума до первого нуля
    aligned_seq1 = []
    aligned_seq2 = []
    i, j = max_pos
    
    # Пока не дойдем до нуля и не выйдем за границы
    while i > 0 and j > 0 and score_matrix[i][j] > 0:
        if trace_matrix[i][j] == 0:  # диагональ
            aligned_seq1.append(seq1[i-1])
            aligned_seq2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif trace_matrix[i][j] == 1:  # сверху (гэп в seq2)
            aligned_seq1.append(seq1[i-1])
            aligned_seq2.append('-')
            i -= 1
        elif trace_matrix[i][j] == 2:  # слева (гэп в seq1)
            aligned_seq1.append('-')
            aligned_seq2.append(seq2[j-1])
            j -= 1
        else:
            # Дошли до нуля или начала
            break
    
    # Разворачиваем, так как шли с конца
    aligned_seq1 = ''.join(reversed(aligned_seq1))
    aligned_seq2 = ''.join(reversed(aligned_seq2))
    
    return aligned_seq1, aligned_seq2, max_score, max_pos, score_matrix

def smith_waterman_affine(seq1, seq2, match_score=2, mismatch_score=-1, 
                          gap_open=-5, gap_extend=-1):
    """
    Реализация алгоритма Смита-Уотермана с аффинными штрафами за гэпы.
    Использует три матрицы: M (main), X (insert), Y (delete)
    """
    
    n = len(seq1) + 1
    m = len(seq2) + 1
    
    # Три матрицы:
    # M - основная (диагональные перемещения - match/mismatch)
    # X - для движений вниз (гэпы в seq2)
    # Y - для движений вправо (гэпы в seq1)
    M = np.zeros((n, m), dtype=int)
    X = np.zeros((n, m), dtype=int)
    Y = np.zeros((n, m), dtype=int)
    
    # Матрицы для трейсинга
    trace_M = np.ones((n, m), dtype=int) * -1
    trace_X = np.ones((n, m), dtype=int) * -1
    trace_Y = np.ones((n, m), dtype=int) * -1
    
    max_score = 0
    max_pos = (0, 0, 'M')  # (i, j, матрица)
    
    # Заполняем матрицы
    for i in range(1, n):
        for j in range(1, m):
            # 1. Основная матрица M (пришли по диагонали)
            # Считаем скор за совпадение/несовпадение
            if seq1[i-1] == seq2[j-1]:
                diag_score = match_score
            else:
                diag_score = mismatch_score
            
            # В M можно прийти из M (продолжаем совпадения), 
            # из X (был гэп в seq2, закрываем его), 
            # из Y (был гэп в seq1, закрываем его)
            candidates = [
                0,  # всегда можно начать новое выравнивание
                M[i-1][j-1] + diag_score,
                X[i-1][j-1] + diag_score,
                Y[i-1][j-1] + diag_score
            ]
            M[i][j] = max(candidates)
            if M[i][j] > 0:
                if M[i][j] == candidates[1]:
                    trace_M[i][j] = 0  # из M по диагонали
                elif M[i][j] == candidates[2]:
                    trace_M[i][j] = 1  # из X по диагонали
                elif M[i][j] == candidates[3]:
                    trace_M[i][j] = 2  # из Y по диагонали
            
            # 2. Матрица X (движение вниз - гэп в seq2)
            # Можно прийти из M (открываем гэп) или из X (продлеваем гэп)
            candidates_X = [
                0,
                M[i-1][j] + gap_open,      # открытие гэпа
                X[i-1][j] + gap_extend     # продление гэпа
            ]
            X[i][j] = max(candidates_X)
            if X[i][j] > 0:
                if X[i][j] == candidates_X[1]:
                    trace_X[i][j] = 0  # открытие из M
                elif X[i][j] == candidates_X[2]:
                    trace_X[i][j] = 1  # продление из X
            
            # 3. Матрица Y (движение вправо - гэп в seq1)
            # Можно прийти из M (открываем гэп) или из Y (продлеваем гэп)
            candidates_Y = [
                0,
                M[i][j-1] + gap_open,      # открытие гэпа
                Y[i][j-1] + gap_extend     # продление гэпа
            ]
            Y[i][j] = max(candidates_Y)
            if Y[i][j] > 0:
                if Y[i][j] == candidates_Y[1]:
                    trace_Y[i][j] = 0  # открытие из M
                elif Y[i][j] == candidates_Y[2]:
                    trace_Y[i][j] = 1  # продление из Y
            
            # Обновляем глобальный максимум
            if M[i][j] > max_score:
                max_score = M[i][j]
                max_pos = (i, j, 'M')
            if X[i][j] > max_score:
                max_score = X[i][j]
                max_pos = (i, j, 'X')
            if Y[i][j] > max_score:
                max_score = Y[i][j]
                max_pos = (i, j, 'Y')
    
    # Трейсинг от максимума
    aligned_seq1 = []
    aligned_seq2 = []
    i, j, matrix_type = max_pos
    
    # Определяем, в какой матрице мы находимся и начинаем трейсинг
    current_matrix = matrix_type
    
    while i > 0 and j > 0:
        if current_matrix == 'M':
            if trace_M[i][j] == 0:  # пришли из M по диагонали
                aligned_seq1.append(seq1[i-1])
                aligned_seq2.append(seq2[j-1])
                i -= 1
                j -= 1
                current_matrix = 'M'
            elif trace_M[i][j] == 1:  # пришли из X по диагонали
                aligned_seq1.append(seq1[i-1])
                aligned_seq2.append(seq2[j-1])
                i -= 1
                j -= 1
                current_matrix = 'X'
            elif trace_M[i][j] == 2:  # пришли из Y по диагонали
                aligned_seq1.append(seq1[i-1])
                aligned_seq2.append(seq2[j-1])
                i -= 1
                j -= 1
                current_matrix = 'Y'
            else:
                break
                
        elif current_matrix == 'X':
            if trace_X[i][j] == 0:  # открытие гэпа из M
                aligned_seq1.append(seq1[i-1])
                aligned_seq2.append('-')
                i -= 1
                current_matrix = 'M'
            elif trace_X[i][j] == 1:  # продление гэпа из X
                aligned_seq1.append(seq1[i-1])
                aligned_seq2.append('-')
                i -= 1
                current_matrix = 'X'
            else:
                break
                
        elif current_matrix == 'Y':
            if trace_Y[i][j] == 0:  # открытие гэпа из M
                aligned_seq1.append('-')
                aligned_seq2.append(seq2[j-1])
                j -= 1
                current_matrix = 'M'
            elif trace_Y[i][j] == 1:  # продление гэпа из Y
                aligned_seq1.append('-')
                aligned_seq2.append(seq2[j-1])
                j -= 1
                current_matrix = 'Y'
            else:
                break
    
    aligned_seq1 = ''.join(reversed(aligned_seq1))
    aligned_seq2 = ''.join(reversed(aligned_seq2))
    
    return aligned_seq1, aligned_seq2, max_score

# Функция для красивого вывода матрицы
def print_matrix(seq1, seq2, matrix, title="Матрица скоров"):
    """Красивый вывод матрицы скоров"""
    print(f"\n{title}:")
    print("    ", " ".join(f"{b:4}" for b in " " + seq2))
    for i, row in enumerate(matrix):
        if i == 0:
            print(f"  {row[0]:4}", " ".join(f"{x:4}" for x in row[1:]))
        else:
            print(f"{seq1[i-1]} {row[0]:4}", " ".join(f"{x:4}" for x in row[1:]))

# Демонстрация на примерах из лекции
def demonstrate_smith_waterman():
    print("=" * 70)
    print("АЛГОРИТМ СМИТА-УОТЕРМАНА (ЛОКАЛЬНОЕ ВЫРАВНИВАНИЕ)")
    print("=" * 70)
    
    # Пример 1: Простой пример с локальным сходством
    print("\n" + "-" * 50)
    print("ПРИМЕР 1: Поиск общего участка")
    print("-" * 50)
    
    seq1 = "ATCGCGTAGC"
    seq2 = "CGCG"
    
    print(f"seq1: {seq1}")
    print(f"seq2: {seq2}")
    
    aligned1, aligned2, score, max_pos, matrix = smith_waterman(seq1, seq2)
    
    print(f"\nЛокальное выравнивание (Смит-Уотерман):")
    print(f"1: {aligned1}")
    print(f"2: {aligned2}")
    print(f"Скор: {score}")
    print(f"Позиция максимума: {max_pos}")
    
    # Сравнение с глобальным выравниванием
    from needleman_wunsch import needleman_wunsch  # предполагаем, что функция доступна
    try:
        global1, global2, global_score, _ = needleman_wunsch(seq1, seq2, 2, -1, -2)
        print(f"\nДля сравнения - глобальное выравнивание (Нидлман-Вунш):")
        print(f"1: {global1}")
        print(f"2: {global2}")
        print(f"Скор: {global_score}")
    except:
        print("\n(Для сравнения с глобальным нужно импортировать needleman_wunsch)")
    
    # Пример 2: Пример из лекции с инверсией
    print("\n" + "-" * 50)
    print("ПРИМЕР 2: Последовательности с общим участком (инверсия)")
    print("-" * 50)
    
    seq1 = "ATCGCG"
    seq2 = "GCGATC"
    
    print(f"seq1: {seq1}")
    print(f"seq2: {seq2}")
    print("Внимание: В seq2 есть участок GCGA, похожий на CGCG в seq1, но в перевернутом виде")
    print("Локальное выравнивание найдет прямые совпадения, но не инверсии")
    
    aligned1, aligned2, score, max_pos, matrix = smith_waterman(seq1, seq2)
    
    print(f"\nЛокальное выравнивание:")
    print(f"1: {aligned1}")
    print(f"2: {aligned2}")
    print(f"Скор: {score}")
    
    # Пример 3: Демонстрация матрицы
    print("\n" + "-" * 50)
    print("ПРИМЕР 3: Демонстрация матрицы скоров")
    print("-" * 50)
    
    seq1 = "ATC"
    seq2 = "ACG"
    
    print(f"seq1: {seq1}")
    print(f"seq2: {seq2}")
    print("Параметры: match=+2, mismatch=-1, gap=-2")
    
    aligned1, aligned2, score, max_pos, matrix = smith_waterman(seq1, seq2, 2, -1, -2)
    
    print_matrix(seq1, seq2, matrix, "Матрица скоров (Смит-Уотерман)")
    print(f"\nМаксимальный скор: {score} в позиции {max_pos}")
    print(f"\nЛокальное выравнивание:")
    print(f"1: {aligned1}")
    print(f"2: {aligned2}")

def demonstrate_affine_penalties():
    """Демонстрация аффинных штрафов за гэпы"""
    print("\n" + "=" * 70)
    print("АФФИННЫЕ ШТРАФЫ ЗА ГЭПЫ")
    print("=" * 70)
    
    # Последовательности, где биологически правильнее иметь один длинный гэп
    seq1 = "AAAAAAA"
    seq2 = "AAA"
    
    print(f"seq1: {seq1}")
    print(f"seq2: {seq2}")
    
    # Сравнение линейного и аффинного штрафов
    print("\n" + "-" * 40)
    print("ЛИНЕЙНЫЙ ШТРАФ (каждый гэп -2):")
    print("-" * 40)
    aligned1, aligned2, score, _, _ = smith_waterman(seq1, seq2, 2, -1, -2)
    print(f"1: {aligned1}")
    print(f"2: {aligned2}")
    print(f"Скор: {score}")
    
    print("\n" + "-" * 40)
    print("АФФИННЫЙ ШТРАФ (открытие -5, продление -1):")
    print("-" * 40)
    aligned1_aff, aligned2_aff, score_aff = smith_waterman_affine(seq1, seq2, 2, -1, -5, -1)
    print(f"1: {aligned1_aff}")
    print(f"2: {aligned2_aff}")
    print(f"Скор: {score_aff}")
    
    print("\n" + "-" * 40)
    print("ОБЪЯСНЕНИЕ:")
    print("-" * 40)
    print("С аффинным штрафом выгоднее сделать один длинный гэп")
    print("(открытие -5 + 4 продления по -1 = всего -9),")
    print("чем четыре коротких гэпа (4 * -5 = -20).")
    print("Это соответствует биологической реальности: вставка/удаление")
    print("одного длинного фрагмента вероятнее, чем нескольких коротких.")

if __name__ == "__main__":
    demonstrate_smith_waterman()
    demonstrate_affine_penalties()
    
    print("\n" + "=" * 70)
    print("КЛЮЧЕВЫЕ ОТЛИЧИЯ СМИТА-УОТЕРМАНА ОТ НИДЛМАНА-ВУНША:")
    print("=" * 70)
    print("1. В формуле появляется 0 → можно начинать выравнивание заново")
    print("2. Ищем максимум не в углу, а во всей матрице")
    print("3. Трейсинг идет от максимума до первого нуля")
    print("4. Результат — только самый похожий участок, без 'хвостов'")
    print("5. Идеально подходит для поиска консервативных доменов")
    print("   или коротких последовательностей в длинных геномах")