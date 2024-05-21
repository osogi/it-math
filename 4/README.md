# Task-4: Метод конечных элементов


## Цель эксперимента
Проверить теоретическую оценку ошибки метода конечных элементов.
## Условия эксперимента
Эксперимент проводился на машине с ОС `Ubuntu 20.04.5 LTS x86_64`, на процессоре `AMD Ryzen 7 4700U`, программа собиралась компилятором `gcc 9.4.0` с флагами указанными в [Makefile](./Makefile)


Код запускался для решения уравнения $y'' - λ*y = -2*λ * sin(\sqrt{λ}*x)$, с граничными условиями $y(0) = 0$, $y(4 * \pi / \sqrt{λ}) = 0$
Решение этого уравнения в явном виде: $y = sin(\sqrt{λ}*x)$
## Теоретические ожидания о результатах эксперимента**

Максимальная ошибка будет того же или меньшего порядка, что и величина $h^2$, где $h \approx 4 * \pi / (\sqrt{λ} * gridSize)$

---

## Результаты эксперимента

Все результаты эксперимента представлены в [этом файле](results.txt).

Замеры проводились только один раз, так как от повторного запуска результат не поменяется, реализация алгоритма полностью детерминирована.

<details>
<summary>Если не хочется лезть в файл</summary>

```bash
./it-math/4$ ./main.elf 1 10
max_error = 0.171182 | h^2 = 1.949551
./it-math/4$ ./main.elf 1 100
max_error = 0.001338 | h^2 = 0.016112
./it-math/4$ ./main.elf 1 1000
max_error = 0.000013 | h^2 = 0.000158
./it-math/4$ ./main.elf 1 10000
max_error = 0.000002 | h^2 = 0.000002

./it-math/4$ ./main.elf 10 10
max_error = 0.171182 | h^2 = 0.194955
./it-math/4$ ./main.elf 10 100
max_error = 0.001338 | h^2 = 0.001611
./it-math/4$ ./main.elf 10 1000
max_error = 0.000013 | h^2 = 0.000016
./it-math/4$ ./main.elf 10 10000
max_error = 0.000001 | h^2 = 0.000000


./it-math/4$ ./main.elf 100 10
max_error = 0.171182 | h^2 = 0.019496
./it-math/4$ ./main.elf 100 100
max_error = 0.001338 | h^2 = 0.000161
./it-math/4$ ./main.elf 100 1000
max_error = 0.000013 | h^2 = 0.000002

./it-math/4$ ./main.elf 1000 10
max_error = 0.171182 | h^2 = 0.001950
./it-math/4$ ./main.elf 1000 100
max_error = 0.001338 | h^2 = 0.000016
./it-math/4$ ./main.elf 1000 1000
max_error = 0.000013 | h^2 = 0.000000
```

</details>

## Оценка полученного результата

Для $λ$ = 1 и 10 теоретические ожидания сбылись, для $λ$ = 100 и 1000 нет. Однако теоретическая оценка предложенная в [книге "Методы вычислений"](http://www.ict.nsc.ru/matmod/files/textbooks/KhakimzyanovCherny-2.pdf) подтверждается, в ней утверждается что максимальная ошибка равняется $O(h^2)$. Если в нашем случае взять $O(h^2) =h^2 * λ/100$, то все ошибки будут ограничены сверху этой границей.