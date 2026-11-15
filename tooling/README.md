# koffient-tooling

Python-инструменты для подготовки и измерения моделей Koffient:

- `bench run` — замеры скорости и памяти на Android-устройстве через adb;
- `quality run`, `quality closeness` — оценка качества модели на компьютере;
- `bench report` — графики и таблицы из записанных результатов.

## Установка

Нужны Python 3.11+, Android platform-tools (`adb`) и переменная `ANDROID_HOME`.

```
python -m venv <папка вне репозитория>
<venv>/Scripts/python -m pip install -e "tooling[dev]"
```

## Замер

```
python -m koffient_tooling bench run --work-dir <рабочая папка> --model <файл.gguf> --build <имя сборки>
python -m koffient_tooling bench show --work-dir <рабочая папка>
```

`--build` — папка сборки llama.cpp (или её имя внутри `<рабочая папка>/build`) с целями
`llama-bench` и `llama-completion`. Раннер сам заливает бинарники и модель на устройство
(повторно не заливает, если контрольная сумма совпала).

Результат — одна строка JSON в `<рабочая папка>/results/runs.jsonl`, сырой вывод инструментов —
в `results/raw/<run_id>/`. Упавший прогон тоже записывается, со `status: failed` и причиной.

## Что измеряется

| Поле | Что это | Чем измерено |
|---|---|---|
| `metrics.prefill` | скорость обработки промпта, ток/с, для каждой длины промпта | `llama-bench` |
| `metrics.decode` | скорость генерации, ток/с | `llama-bench` |
| `metrics.ttft_ms` | время до первого токена без загрузки модели: обработка промпта из `params.ttft_prompt_tokens` токенов и выбор первого токена | `llama-completion` |
| `metrics.load_ms` | время от старта загрузки до конца прогревочного прогона | `llama-completion` |
| `metrics.peak_rss_mb` | пик резидентной памяти процесса при контексте `params.n_ctx` | `time -v` |

`warm` — файл модели уже в кэше ОС, `cold` — кэш сброшен перед запуском (нужен root; без него `cold` пуст).
Мегабайты везде десятичные (10^6 байт).

Кроме замеров, запись содержит паспорт модели (контрольная сумма, фактические типы тензоров,
бит на параметр), паспорт сборки движка (коммит, флаги CMake, набор инструкций, который движок
реально использует) и описание устройства.

## Ограничения

- На эмуляторе скорость не телефонная: в записи стоит `device.speed_is_phone_grade: false`.
  Такие цифры годятся только для сравнения вариантов между собой на одном и том же эмуляторе.
- Холодная загрузка на эмуляторе читает с виртуального диска хоста и тоже не телефонная.

## Качество

Качество считается на компьютере, а не на устройстве: тем же файлом модели и официальной сборкой
llama.cpp той же версии. Результат — строки в `<рабочая папка>/results/quality.jsonl`, ответы модели
по каждому примеру — в `results/raw/quality/<eval_id>/answers.jsonl`.

```
python -m koffient_tooling quality run --work-dir <рабочая папка> --model <файл.gguf>
python -m koffient_tooling quality closeness --work-dir <рабочая папка> --model <файл.gguf> --base <fp16.gguf>
python -m koffient_tooling quality show --work-dir <рабочая папка>
```

- `quality run` — быстрый набор: 11 задач, одна запись на задачу.
- `quality closeness` — близость к fp16: perplexity, KL-расхождение и доля совпадений первого токена
  на фиксированном английском и русском тексте (`llama-perplexity`). Логиты fp16 сохраняются один раз
  в `results/kld/`, около 3 ГБ на язык.
- `quality show` — таблица «задача × вариант модели» с долей ответов, изменившихся относительно базовой
  модели.

### Быстрый набор

| Задача | Сценарий | Роль | Примеров | Оценка |
|---|---|---|---|---|
| `bfcl-en`, `bfcl-ru` | вызов функций | основной | 300 | сравнение вызова с эталоном (проверяющий код BFCL) |
| `jsonschema` | JSON по схеме | основной | 200 | ответ проходит валидацию по схеме |
| `ifeval-en`, `ifeval-ru` | следование инструкциям | основной | 300 | правила IFEval, строгий режим |
| `samsum-en`, `samsum-ru` | суммаризация переписки | вторичный | 200 | ROUGE-L |
| `mgsm-en`, `mgsm-ru` | математика | контрольный | 250 | совпадение итогового числа |
| `mmlu-en`, `mmlu-ru` | знания | контрольный | 285 | выбор буквы A–D |

Подвыборки фиксированы и зависят только от идентификатора примера, поэтому в парных задачах
английский и русский варианты содержат одни и те же примеры.

### Почему несколько серверов по одному запросу

Ответы генерирует пул независимых `llama-server`, у каждого один слот. Если дать одному серверу несколько
слотов, запросы считаются одним пакетом, и ответ на запрос начинает зависеть от того, какие ещё запросы
оказались в пакете: на одной и той же выборке точность менялась от прогона к прогону. С пулом ответы
повторяются точно. Они всё равно зависят от числа потоков, поэтому `--threads` менять между сравниваемыми
вариантами нельзя.

Пул упирается в пропускную способность памяти: на 16 ядрах около 170 ток/с суммарно для файла 0,5 ГБ.

### Что должно лежать в рабочей папке

| Путь | Что это |
|---|---|
| `host/llama-b12199-cpu-x64/` | официальная сборка llama.cpp под Windows, архив `llama-b12199-bin-win-cpu-x64.zip` из релиза `b12199` |
| `third_party/rfcb/` | `git clone https://github.com/sir-timio/RFCB`, коммит `a6b4896`: данные BFCL v3 на двух языках и проверяющий код |
| `third_party/ifeval/` | `git clone https://github.com/oKatanaaa/ifeval`, коммит `ac8c1f2`; установить `pip install third_party/ifeval` |
| `nltk_data/` | `python -c "import nltk; nltk.download('punkt', download_dir='<рабочая папка>/nltk_data')"` |
| `datasets/` | наборы данных, см. ниже |

Наборы скачиваются с Hugging Face в указанных ревизиях (`https://huggingface.co/datasets/<набор>/resolve/<ревизия>/<файл>`):

| Папка и файл в `datasets/` | Набор, ревизия, исходный файл |
|---|---|
| `jsonschemabench/Glaiveai2K-test.parquet`, `Github_easy-test.parquet` | `epfl-dlab/JSONSchemaBench`, `5bd0f46`, `<часть>/test-00000-of-00001.parquet` |
| `ifeval/en.jsonl`, `ru.jsonl` | `kaleinaNyan/instruction-following-eval`, `843295a` |
| `samsum/test.csv` | `knkarthick/samsum`, `6b929ff` |
| `samsum-ru/test-*.parquet` | `d0rj/samsum-ru`, `1793e9d`, `data/test-*.parquet` |
| `mgsm/en-test.parquet`, `ru-test.parquet` | `juletxara/mgsm`, `b2f13d4`, `<язык>/test-00000-of-00001.parquet` |
| `mmlu/test.parquet` | `cais/mmlu`, `c30699e`, `all/test-00000-of-00001.parquet` |
| `rummlu/test.jsonl` | `MERA-evaluation/MERA`, `15c0e5b`, `data/rummlu/test.jsonl` |
| `corpus/en.txt` | `wiki.test.raw` из `wikitext-2-raw-v1.zip` (`ggml-org/ci`) |
| `corpus/ru.txt` | первые 40 статей `wikimedia/wikipedia`, конфигурация `20231101.ru`, по 6000 символов каждой |

Наборы в репозиторий не кладём: у части из них нет лицензии или она запрещает производные работы.
Они используются только для внутренних замеров.

Зависимости: `pip install -e "tooling[dev,quality,report]"`.

## Отчёт с графиками

```
python -m koffient_tooling bench report --work-dir <рабочая папка> --name <тема>
```

Строит из `runs.jsonl` и `quality.jsonl` папку `results/reports/<дата>-<тема>/`: графики PNG
(качество против размера, близость к fp16, изменившиеся ответы, скорость и память), те же числа
таблицами в `report.md` и идентификаторы исходных записей в `data.json`. Папка отчёта не меняется
при появлении новых прогонов: на неё можно ссылаться из журнала решений.

## Тесты

```
cd tooling && python -m pytest
```

Тесты проверяют разбор вывода и ответов на образцах из реальных прогонов (`tests/fixtures`);
устройство, модель и наборы данных не нужны.
