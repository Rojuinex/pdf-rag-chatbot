FROM python:3.12

# Install tmux
RUN apt update && \
	apt install tmux && \
	rm -rf /var/lib/apt/lists/*

# Install neovim and kickstart
RUN curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux64.tar.gz && \
	rm -rf /opt/nvim && \
	tar -C /opt -xzf nvim-linux64.tar.gz && \
	rm -rf nvim-linux64.tar.gz && \
	echo export PATH="$PATH:/opt/nvim-linux64/bin" >> "/root/.bashrc}" && \
	git clone https://github.com/nvim-lua/kickstart.nvim.git "/root/.config}"/nvim

# PDM
RUN curl -sSL https://pdm-project.org/install-pdm.py | python3 - && \
	echo 'export PATH="$PATH:root.local/bin"' >> "/root/.bashrc"
