from isaacsim import SimulationApp

app = SimulationApp({"headless": False})

from omni.isaac.core.utils.stage import open_stage

open_stage("C:/Users/aaron/Documents/repositories/robo-garden/workspace/robots/urchin_v2/assets/urchin_v2.usd")

while app.is_running():
    app.update()

app.close()
