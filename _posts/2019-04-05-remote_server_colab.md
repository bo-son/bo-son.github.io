---
published: true
layout: single
title: "Colab에서 remote server와 local runtime 연결하기"
path: "2019-04-05-remote_server_colab"
use_math: true
category: "Tips"
---

Google colab에서 로컬 런타임으로 **외부 서버**에 연결하려고 하는 경우, [공식 documentation](https://research.google.com/colaboratory/local-runtimes.html)대로 하면 "Unable to connect to the runtime" 에러가 뜨는 경우가 있다(많다). 아래의 방법으로 하니 안정적으로 연결이 되었다.

일단 공식 docs의 step 2를 따라 jupyter_http_over_ws 세팅을 한다.
```
pip install jupyter_http_over_ws && \
jupyter serverextension enable --py jupyter_http_over_ws
```

외부 서버에서 아래 설정을 따라 노트북 서버를 연다.
```
jupyter notebook --no-browser \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=PORT \
  --NotebookApp.token='' --NotebookApp.password='' \
  --NotebookApp.disable_check_xsrf=True \
  --NotebookApp.port_retries=0
```

개인 기기(랩탑)으로 포워딩한다. PORT는 같은 번호로 하는 게 안전한 것 같다.
```
ssh MYID@147.46.3.25 -L -f PORT:localhost:PORT -N
```

Chrome에서 http://localhost:PORT 접속 후 colab에 들어가서 로컬 런타임에 연결(backend 포트를 PORT로 설정)한다. Safari에서는 연결이 안 되는 경우가 많은데 웹소켓 문제가 아닐까 싶다.