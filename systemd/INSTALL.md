# Installing the daily timer

These are **user units** — they run as your normal user, no `sudo` for the
service itself, and they survive reboots.

## One-time setup

```bash
# 1. Make sure run_daily.sh is executable (already done in the repo)
chmod +x ~/tradeInside/run_daily.sh

# 2. Drop the unit files into your user systemd directory
mkdir -p ~/.config/systemd/user
cp ~/tradeInside/systemd/tradeinside-pipeline.service ~/.config/systemd/user/
cp ~/tradeInside/systemd/tradeinside-pipeline.timer   ~/.config/systemd/user/

# 3. Reload systemd, enable the timer, start it
systemctl --user daemon-reload
systemctl --user enable --now tradeinside-pipeline.timer

# 4. (Optional but important on a desktop that you log out of) — let your
#    user units keep running when you're not logged in
sudo loginctl enable-linger "$USER"
```

## Verify it's scheduled

```bash
systemctl --user list-timers tradeinside-pipeline.timer
# NEXT                      LAST  LEFT      UNIT                             ACTIVATES
# Mon 2025-05-05 07:01:23   -     12h left  tradeinside-pipeline.timer       tradeinside-pipeline.service
```

## Run it once right now (don't wait until 07:00 tomorrow)

```bash
systemctl --user start tradeinside-pipeline.service
```

## See the logs

```bash
# stream the most recent run
journalctl --user -u tradeinside-pipeline.service -n 200 --no-pager

# follow live
journalctl --user -u tradeinside-pipeline.service -f

# or just tail the per-run file
tail -200 ~/tradeInside/logs/last.log
```

## Change the schedule

Edit `~/.config/systemd/user/tradeinside-pipeline.timer`, change the
`OnCalendar=` line, then:

```bash
systemctl --user daemon-reload
systemctl --user restart tradeinside-pipeline.timer
```

`OnCalendar=` syntax cheat sheet:
- `Mon..Fri 07:00`           — current default
- `daily`                    — every day at 00:00 local
- `*-*-* 22:00:00`           — every day at 22:00 local
- `Mon..Fri 07:00,18:00`     — twice a day, weekdays

## Stop / disable

```bash
systemctl --user stop tradeinside-pipeline.timer
systemctl --user disable tradeinside-pipeline.timer
```

## Cron alternative (if you'd rather not use systemd)

Add this single line to `crontab -e` instead:

```
0 7 * * 1-5  cd $HOME/tradeInside && ./run_daily.sh
```

You'll lose `Persistent=true` (missed runs don't catch up) and the per-step
journald logs, but it's one line and works everywhere.
