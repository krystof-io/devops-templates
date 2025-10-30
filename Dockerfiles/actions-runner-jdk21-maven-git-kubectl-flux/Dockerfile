# Define base image version
# https://github.com/actions/runner/releases
# https://github.com/actions/runner/pkgs/container/actions-runner
ARG RUNNER_VERSION=latest
FROM ghcr.io/actions/actions-runner:${RUNNER_VERSION}

# Define versions for easy maintenance
# https://adoptium.net/temurin/releases
# Temurin version:  21.0.8+9
# Package version:  21.0.8.0.0+9-0
#                   │  │ │ │ │ │ └─ Package revision (Debian/Ubuntu packaging)
#                   │  │ │ │ │ └─── Build number (from Temurin)
#                   │  │ │ │ └───── Extra field (unused, set to 0)
#                   │  │ │ └─────── Micro version (unused, set to 0)
#                   │  │ └───────── Patch version (8)
#                   │  └─────────── Minor version (0)
#                   └────────────── Major version (21)
ARG JDK_VERSION=21.0.8.0.0+9-0
# https://maven.apache.org/download.cgi
ARG MAVEN_VERSION=3.9.11
# https://github.com/mikefarah/yq/releases
ARG YQ_VERSION=4.48.1
# https://github.com/kubernetes/kubernetes/releases
# https://dl.k8s.io/release/stable.txt
ARG KUBECTL_VERSION=1.34.1
# https://github.com/fluxcd/flux2/releases
ARG FLUX_VERSION=2.7.2
# https://github.com/cli/cli/releases
ARG GH_CLI_VERSION=2.82.1

# Switch to root to install packages
USER root

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    apt-transport-https \
    ca-certificates \
    software-properties-common \
    jq \
    unzip \
    && rm -rf /var/lib/apt/lists/*


# Install JDK 21 from Adoptium (pinned version for reproducibility)
RUN wget -qO /usr/share/keyrings/adoptium.asc https://packages.adoptium.net/artifactory/api/gpg/key/public || exit 1 && \
    echo "deb [signed-by=/usr/share/keyrings/adoptium.asc] https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | tee /etc/apt/sources.list.d/adoptium.list && \
    apt-get update && \
    apt-get install -y temurin-21-jdk=${JDK_VERSION} && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    java -version && javac -version

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Set Maven environment variables
ENV MAVEN_HOME=/usr/local/maven
ENV PATH=$PATH:$MAVEN_HOME/bin

# Install Maven (pinned version for reproducibility)
RUN wget https://archive.apache.org/dist/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz && \
    tar xzf apache-maven-${MAVEN_VERSION}-bin.tar.gz && \
    mv apache-maven-${MAVEN_VERSION} /usr/local/maven && \
    rm apache-maven-${MAVEN_VERSION}-bin.tar.gz && \
    mvn -version

# Install yq (pinned version for reproducibility)
RUN BINARY=yq_linux_amd64 && \
    wget https://github.com/mikefarah/yq/releases/download/v${YQ_VERSION}/${BINARY}.tar.gz -O - | \
    tar xz && \
    mv ${BINARY} /usr/bin/yq && \
    yq --version

# Install kubectl (pinned version for reproducibility)
RUN wget https://dl.k8s.io/release/v${KUBECTL_VERSION}/bin/linux/amd64/kubectl && \
    chmod +x kubectl && \
    mv kubectl /usr/local/bin/ && \
    kubectl version --client=true

# Install Flux CLI (pinned version for reproducibility)
RUN wget https://github.com/fluxcd/flux2/releases/download/v${FLUX_VERSION}/flux_${FLUX_VERSION}_linux_amd64.tar.gz && \
    tar xzf flux_${FLUX_VERSION}_linux_amd64.tar.gz && \
    mv flux /usr/local/bin/ && \
    rm flux_${FLUX_VERSION}_linux_amd64.tar.gz && \
    flux --version

# Install GitHub CLI (pinned version for reproducibility)
RUN wget https://github.com/cli/cli/releases/download/v${GH_CLI_VERSION}/gh_${GH_CLI_VERSION}_linux_amd64.tar.gz && \
    tar xzf gh_${GH_CLI_VERSION}_linux_amd64.tar.gz && \
    mv gh_${GH_CLI_VERSION}_linux_amd64/bin/gh /usr/local/bin/ && \
    rm -rf gh_${GH_CLI_VERSION}_linux_amd64* && \
    gh --version

# Switch back to the default user
USER runner
