# Student Kit: Cloud Computing Mini Project

Read the project handout first. It explains every file here.

    data/commands_pool.jsonl     3,000 commands (text only)
    data/validation.jsonl        100 labelled commands, for evaluation only (never train on them)
    data/schema.json             the four keys, the eight intents and the slot rules
    data/rate_card.json          hypothetical prices for the cost exercise
    tools/score.py               python tools/score.py PREDICTIONS data/validation.jsonl [--min-validity 95]
    tools/locustfile.py          pip install locust, then:
                                 locust -f tools/locustfile.py --headless --users 4 --spawn-rate 4 --run-time 2m --host http://localhost:8080 --csv results/loadtest
    tools/nacl_sim.py            python tools/nacl_sim.py network/nacl.json network/packets_sample.jsonl --check
    network/packets_sample.jsonl six sample packets with expected decisions

score.py and nacl_sim.py need only Python 3 (standard library); run them with -h for help. The load test needs Locust (https://locust.io).

Data: commands from SLURP (Bastianelli et al., EMNLP 2020), licensed CC BY 4.0.
https://github.com/pswietojanski/slurp
