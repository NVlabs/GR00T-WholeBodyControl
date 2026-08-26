# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac

# don't put duplicate lines or lines starting with space in the history.
# See bash(1) for more options
HISTCONTROL=ignoreboth

# append to the history file, don't overwrite it
shopt -s histappend

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=1000
HISTFILESIZE=2000

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize

# If set, the pattern "**" used in a pathname expansion context will
# match all files and zero or more directories and subdirectories.
#shopt -s globstar

# make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "${debian_chroot:-}" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
    xterm-color|*-256color) color_prompt=yes;;
esac

# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
	# We have color support; assume it's compliant with Ecma-48
	# (ISO/IEC-6429). (Lack of such support is extremely rare, and such
	# a case would tend to support setf rather than setaf.)
	color_prompt=yes
    else
	color_prompt=
    fi
fi

if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt

# If this is an xterm set the title to user@host:dir
case "$TERM" in
xterm*|rxvt*)
    PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
    ;;
*)
    ;;
esac

# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# colored GCC warnings and errors
export GCC_COLORS='error=01;31:warning=01;35:note=01;36:caret=01;32:locus=01:quote=01'

# Set terminal type for color support
export TERM=xterm-256color

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# Add an "alert" alias for long running commands.  Use like so:
#   sleep 10; alert
alias alert='notify-send --urgency=low -i "$([ $? = 0 ] && echo terminal || echo error)" "$(history|tail -n1|sed -e '\''s/^\s*[0-9]\+\s*//;s/[;&|]\s*alert$//'\'')"'

# Alias definitions.
# You may want to put all your additions into a separate file like
# ~/.bash_aliases, instead of adding them here directly.
# See /usr/share/doc/bash-doc/examples in the bash-doc package.

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi

# useful commands
bind '"\e[A": history-search-backward'
bind '"\e[B": history-search-forward'

# Store the last 10 directories in a history file
CD_HISTFILE=~/.cd_history
CD_HISTSIZE=10

cd() {
    local histfile="${CD_HISTFILE:-$HOME/.cd_history}"
    local max="${CD_HISTSIZE:-10}"

    case "$1" in
        --)   [ -f "$histfile" ] && tac "$histfile" | nl -w2 -s' ' || echo "No directory history yet."; return ;;
        -[0-9]*)
            local idx=${1#-}
            local dir=$(tac "$histfile" 2>/dev/null | sed -n "${idx}p")
            [ -n "$dir" ] && builtin cd "$dir" || echo "Invalid selection: $1"
            return ;;
    esac

    builtin cd "$@" || return

    [[ $(tail -n1 "$histfile" 2>/dev/null) != "$PWD" ]] && echo "$PWD" >> "$histfile"
    tail -n "$max" "$histfile" > "${histfile}.tmp" && mv "${histfile}.tmp" "$histfile"
}

# Make decoupled_wbc importable
export PYTHONPATH="${DECOUPLED_WBC_DIR}:${PYTHONPATH}"

# Manus to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$DECOUPLED_WBC_DIR/decoupled_wbc/control/teleop/device/SDKClient_Linux/ManusSDK/lib:$LD_LIBRARY_PATH

# CUDA support
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib:/lib/x86_64-linux-gnu:/lib64:/lib:$LD_LIBRARY_PATH 

# decoupled_wbc aliases
alias dg="python $DECOUPLED_WBC_DIR/decoupled_wbc/scripts/deploy_g1.py"
alias rsl="python $DECOUPLED_WBC_DIR/decoupled_wbc/control/main/teleop/run_sim_loop.py"
alias rgcl="python $DECOUPLED_WBC_DIR/decoupled_wbc/control/main/teleop/run_g1_control_loop.py"
alias rtpl="python $DECOUPLED_WBC_DIR/decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py"
alias tgcl="pytest $DECOUPLED_WBC_DIR/decoupled_wbc/tests/control/main/teleop/test_g1_control_loop.py -s"
