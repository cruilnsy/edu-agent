# frontend/Dockerfile
# Builds the Vue app and serves it with nginx, which also forwards /api to the backend.
#     docker build -t eduagent-frontend frontend

FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
