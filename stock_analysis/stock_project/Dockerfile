FROM python:3.9-slim
RUN sed -i 's|http://deb.debian.org|https://mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources
RUN sed -i 's|http://security.debian.org|https://mirrors.aliyun.com/debian-security|g' /etc/apt/sources.list.d/debian.sources
WORKDIR /stock_project/
COPY ./requirements /stock_project/requirements.txt
COPY ./backend_start.sh /stock_project/backend_start.sh
# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update --fix-missing && apt-get upgrade -y
RUN apt-get install build-essential -y
RUN apt-get install default-libmysqlclient-dev -y
RUN apt-get install pkg-config -y
RUN pip3 config set global.index-url http://mirrors.aliyun.com/pypi/simple/
RUN pip3 config set install.trusted-host mirrors.aliyun.com
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt
# 一次性安装所有Python依赖
RUN pip3 install pyautogui pyperclip pillow pytesseract mysql-connector-python pygetwindow opencv-python httpx
RUN pip3 install mysqlclient ta requests akshare
RUN pip3 install tushare aiomysql
EXPOSE 8010
RUN chmod +x /stock_project/backend_start.sh
CMD ["/stock_project/backend_start.sh"]