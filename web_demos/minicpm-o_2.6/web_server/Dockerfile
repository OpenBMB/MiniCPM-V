# FROM 基于node的版本镜像，并通过构建阶段命名，将有node环境的阶段命名为build-stage
FROM modelbest-registry-vpc.cn-beijing.cr.aliyuncs.com/modelbest/playground:20.10.0 as build-stage
# 设置工作区为 /build 于系统文件隔离
WORKDIR /build
COPY . /build

# 在容器中安装依赖
RUN npm config set registry https://registry.npmmirror.com/ 
# 或者用源 https://registry.npm.taobao.org
RUN npm i pnpm -g
RUN pnpm config set registry https://registry.npmmirror.com/
RUN pnpm install

# 打包
RUN pnpm run build

# production stage
FROM modelbest-registry-vpc.cn-beijing.cr.aliyuncs.com/modelbest/playground:alpine as production-stage
COPY --from=build-stage /build/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/
EXPOSE 3000