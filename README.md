# Trabalho Final de DAS

Trabalho final da disciplina de Desenvolvimento Avançado de Software da UnB.

Professor: Teófilo Campos.

Alunas: Ana Carolina e Priscilla.

## Dependências

* [OPENCV](http://opencv.org/downloads.html)

* [CAFFE](http://caffe.berkeleyvision.org/installation.html)

* Depois instale o Django na versão 1.10.3 Uma opção é executar a instalação por meio do _pip_.

```
pip install django
```

* E também o [BOOTSTRAP3](http://django-bootstrap3.readthedocs.io/en/latest/installation.html)

```
pip install django-bootstrap3
```

## Guia de Instalação

* Clone o repositório:

```
git clone https://github.com/anacarolina1/identificadordeimagens.git
```

* Abra em identificadordeimagens/image_retrieval_web e execute o comandos abaixo:

```
python manage.py makemigrations

python manage.py migrate
```

* A aplicação é executada por meio do comando abaixo:

```
python manage.py runserver
```
