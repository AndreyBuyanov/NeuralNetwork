FROM rikorose/gcc-cmake:gcc-10

COPY . /app

RUN cmake -S app -B app/build \
    && cmake --build app/build --config RelWithDebInfo \
    && rm -rf app/build

CMD /app/bin/AppDigits
