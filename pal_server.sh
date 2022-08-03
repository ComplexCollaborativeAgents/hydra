cd ./bin/pal
# ./gradlew --no-daemon --stacktrace runclient
xvfb-run -s '-screen 0 1280x1024x24' ./gradlew --no-daemon --stacktrace runclient
