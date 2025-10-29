FROM node:18-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine

# Add nginx user with UID 2000
RUN addgroup -g 2000 nginx-group && \
    adduser -u 2000 -G nginx-group -s /bin/sh -D nginx-user && \
    mkdir -p /tmp/nginx/temp /tmp/nginx/conf && \
    chown -R nginx-user:nginx-group /tmp/nginx

# Install envsubst
RUN apk add --no-cache gettext

# Copy the built app
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/templates/default.conf.template

# Switch to UID 2000
USER 2000

ENV BACKEND_URL=localhost
EXPOSE 8080

CMD ["sh", "-c", "envsubst '${BACKEND_URL}' < /etc/nginx/templates/default.conf.template > /tmp/nginx/conf/nginx.conf && nginx -c /tmp/nginx/conf/nginx.conf -g 'daemon off;'"]