import os
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import robocasa
import robocasa.macros as macros

SINGLE_STAGE_TASK_DATASETS = OrderedDict(
    PnPCounterToCab=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24",
        mg_path="v0.1/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-05-04-22-12-27_and_2024-05-07-07-39-33",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/c0a0vdyqeqh6o9z4c57sk61aypladizj.hdf5",
            human_im="https://utexas.box.com/shared/static/gznii250ip99731ii2r0ml6be7gzt2cp.hdf5",
            mg_im="https://utexas.box.com/shared/static/7y9csrcx6uhhq4p3yctmm2df3rjqpw6g.hdf5",
        ),
    ),
    PnPCabToCounter=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24",
        mg_path="v0.1/single_stage/kitchen_pnp/PnPCabToCounter/mg/2024-07-12-04-33-29",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/fgd165p91ts9qurfb496ubi1xdqgrz97.hdf5",
            human_im="https://utexas.box.com/shared/static/pdfh88slq4bazfglixaw80whvbtlbw6r.hdf5",
            mg_im="https://utexas.box.com/shared/static/mfltre5w8qi8ps1fqicblx40p008425z.hdf5",
        ),
    ),
    PnPCounterToSink=dict(
        horizon=700,
        human_path="v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25",
        mg_path="v0.1/single_stage/kitchen_pnp/PnPCounterToSink/mg/2024-05-04-22-14-06_and_2024-05-07-07-40-17",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/nc0x55xwlgs4acj97ngdj1q6pv450ooo.hdf5",
            human_im="https://utexas.box.com/shared/static/0z1wr8iwfusfqqpp56j3vfxvvse1x0vc.hdf5",
            mg_im="https://utexas.box.com/shared/static/5vkb2fpthnw554q888fynk3zqwsgk2e5.hdf5",
        ),
    ),
    PnPSinkToCounter=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2",
        mg_path="v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/uyhepwjyr0880a5q03fc61qmhbrhwy7k.hdf5",
            human_im="https://utexas.box.com/shared/static/pe03k5qvcuq1nzbpfag5pfo2t8fzvl9i.hdf5",
            mg_im="https://utexas.box.com/shared/static/ppj3j8jnxkpusi7ft44f5ut7szkmu8yr.hdf5",
        ),
    ),
    PnPCounterToMicrowave=dict(
        horizon=600,
        human_path="v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-27",
        mg_path="v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/mg/2024-05-04-22-13-21_and_2024-05-07-07-41-17",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/z1qwxsuczjmv68p7267nylbv9ebc6g4p.hdf5",
            human_im="https://utexas.box.com/shared/static/2kun70ehqzanl5h0hbgpdezrfrvuimq8.hdf5",
            mg_im="https://utexas.box.com/shared/static/vdm6pzl68xqqxvk9iy7i3l62j37t3sbp.hdf5",
        ),
    ),
    PnPMicrowaveToCounter=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/2024-04-26",
        mg_path="v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/mg/2024-05-04-22-14-26_and_2024-05-07-07-41-42",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/emnl5i3s621sf5smgek5lu4twbc326jt.hdf5",
            human_im="https://utexas.box.com/shared/static/z2t1kyto32thprw7xvq3r4tjvkdz0dl2.hdf5",
            mg_im="https://utexas.box.com/shared/static/3gs5cvp5qyzvk7gplmp0lkwfs1x1mkrj.hdf5",
        ),
    ),
    PnPCounterToStove=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26",
        mg_path="v0.1/single_stage/kitchen_pnp/PnPCounterToStove/mg/2024-05-04-22-14-20",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/9z2k5iqspwdo2153yuutcvqb9zzzvtdu.hdf5",
            human_im="https://utexas.box.com/shared/static/penrv5sysn192gxwj7n7f6k0gjceovfn.hdf5",
            mg_im="https://utexas.box.com/shared/static/1ytpoks6mp9f56wh36ni8k02fz0kbvmb.hdf5",
        ),
    ),
    PnPStoveToCounter=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01",
        mg_path="v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/mg/2024-05-04-22-14-40",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/1m3v1rkh7gna5ujt0vcwzej7stv5rdio.hdf5",
            human_im="https://utexas.box.com/shared/static/rht9bfh38gaue5ig17ubpsibb4jrgtyv.hdf5",
            mg_im="https://utexas.box.com/shared/static/uvrx2d7zxungplafbn9kg6rbe23ysund.hdf5",
        ),
    ),
    OpenSingleDoor=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24",
        mg_path="v0.1/single_stage/kitchen_doors/OpenSingleDoor/mg/2024-05-04-22-37-39",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/hjukxdpqvnrghtbjbhin28idzkysq7oa.hdf5",
            human_im="https://utexas.box.com/shared/static/vhvlbuza3g1lpqa9m6zvmbzh8xquvqy8.hdf5",
            mg_im="https://utexas.box.com/shared/static/ekf6qwp9s07jnkpazbohlv03d6b73vth.hdf5",
        ),
    ),
    CloseSingleDoor=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_doors/CloseSingleDoor/2024-04-24",
        mg_path="v0.1/single_stage/kitchen_doors/CloseSingleDoor/mg/2024-05-04-22-34-56",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/gea28p5mrvj9lnw1s6c8fqvzxk5f779i.hdf5",
            human_im="https://utexas.box.com/shared/static/2wnm0u1x9fp9ni02pmzqzhpjsb1kgfrr.hdf5",
            mg_im="https://utexas.box.com/shared/static/5mg9vkhilkaxza83l48aht3sk76086tm.hdf5",
        ),
    ),
    OpenDoubleDoor=dict(
        horizon=1000,
        human_path="v0.1/single_stage/kitchen_doors/OpenDoubleDoor/2024-04-26",
        mg_path="v0.1/single_stage/kitchen_doors/OpenDoubleDoor/mg/2024-05-04-22-35-53",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/xl23utiszzdbjdqlveq1mxxca9m3kbvw.hdf5",
            human_im="https://utexas.box.com/shared/static/8swihowjd5fdk1vpf0k94gl72f8nbjeb.hdf5",
            mg_im="https://utexas.box.com/shared/static/5djutjyyksic6qpl2uuq9cvzj3r78jvf.hdf5",
        ),
    ),
    CloseDoubleDoor=dict(
        horizon=700,
        human_path="v0.1/single_stage/kitchen_doors/CloseDoubleDoor/2024-04-29",
        mg_path="v0.1/single_stage/kitchen_doors/CloseDoubleDoor/mg/2024-05-04-22-22-42_and_2024-05-08-06-02-36",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/zozyctzejc2nrlpjwqjq4w761830kq45.hdf5",
            human_im="https://utexas.box.com/shared/static/14f2ssfhwfhyo9cvj0s3kq303bv38g6t.hdf5",
            mg_im="https://utexas.box.com/shared/static/7t269kmcrsycc4wz5wonw62ijopjhyv7.hdf5",
        ),
    ),
    OpenDrawer=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_drawer/OpenDrawer/2024-05-03",
        mg_path="v0.1/single_stage/kitchen_drawer/OpenDrawer/mg/2024-05-04-22-38-42",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/990d2f339zvalvlw50s4s1l6sfvt018m.hdf5",
            human_im="https://utexas.box.com/shared/static/d8a0g5827kbm4ufk3p1tbuz3idstgum8.hdf5",
            mg_im="https://utexas.box.com/shared/static/t0sio5vrdax5bdfqkumfh63inq6o9b88.hdf5",
        ),
    ),
    CloseDrawer=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30",
        mg_path="v0.1/single_stage/kitchen_drawer/CloseDrawer/mg/2024-05-09-09-32-19",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/ooekd1zdy02hfu234xm5h63f7mzbez4b.hdf5",
            human_im="https://utexas.box.com/shared/static/4r5w0a6i4jtgv5qmqx09fnqh5d7c45oi.hdf5",
            mg_im="https://utexas.box.com/shared/static/aohabqltp8c6ze61u4h2uhtc9p9w35zb.hdf5",
        ),
    ),
    TurnOnSinkFaucet=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet/2024-04-25",
        mg_path="v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet/mg/2024-05-04-22-17-46",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/7vogk85ed3sm1o1vo8fzast9rs8zshgk.hdf5",
            human_im="https://utexas.box.com/shared/static/f0brygtzwgmmwccg58b6m82yapa4jnji.hdf5",
            mg_im="https://utexas.box.com/shared/static/p8x9njpsf5j9fkmnu7lwajupzm1t7aap.hdf5",
        ),
    ),
    TurnOffSinkFaucet=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_sink/TurnOffSinkFaucet/2024-04-25",
        mg_path="v0.1/single_stage/kitchen_sink/TurnOffSinkFaucet/mg/2024-05-04-22-17-26",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/ukkycoa2c6k5d6bpt1cdplzk2nrg879t.hdf5",
            human_im="https://utexas.box.com/shared/static/ceewfn4ydhprupdcdppfe8wu4x61oxdg.hdf5",
            mg_im="https://utexas.box.com/shared/static/r392ma0dje2t5ov4dug4vbqhn0z6hblk.hdf5",
        ),
    ),
    TurnSinkSpout=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_sink/TurnSinkSpout/2024-04-29",
        mg_path="v0.1/single_stage/kitchen_sink/TurnSinkSpout/mg/2024-05-09-09-31-12",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/zxop70u9s6rl4udz7fhp4besrvu508pj.hdf5",
            human_im="https://utexas.box.com/shared/static/nnojt3760k143fxwhf2u0kzh1i56n7x1.hdf5",
            mg_im="https://utexas.box.com/shared/static/34l1y45xquau66nk8j7hrpfvlad4b9fg.hdf5",
        ),
    ),
    TurnOnStove=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_stove/TurnOnStove/2024-05-02",
        mg_path="v0.1/single_stage/kitchen_stove/TurnOnStove/mg/2024-05-08-09-20-31",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/ow4npc933yty4blxg8rc2pgjp6f1yc36.hdf5",
            human_im="https://utexas.box.com/shared/static/dewbngq5wk6dipb29x6984x6dygu2gck.hdf5",
            mg_im="https://utexas.box.com/shared/static/7gzolvj3fyzhhfjoxswmmpjpxu5q4too.hdf5",
        ),
    ),
    TurnOffStove=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_stove/TurnOffStove/2024-05-02",
        mg_path="v0.1/single_stage/kitchen_stove/TurnOffStove/mg/2024-05-08-09-20-45",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/vp6u3huv9c2vg5r09105gkbmwguycnn0.hdf5",
            human_im="https://utexas.box.com/shared/static/8tukea09szcjb2ncbe43zt65n7dzg4eh.hdf5",
            mg_im="https://utexas.box.com/shared/static/6rv14y77hfknrwfbwn5qtlfdydkr8tog.hdf5",
        ),
    ),
    CoffeeSetupMug=dict(
        horizon=600,
        human_path="v0.1/single_stage/kitchen_coffee/CoffeeSetupMug/2024-04-25",
        mg_path="v0.1/single_stage/kitchen_coffee/CoffeeSetupMug/mg/2024-05-04-22-22-13_and_2024-05-08-05-52-13",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/zxytzos86xes7jo8z0gp7lklawbiyo8g.hdf5",
            human_im="https://utexas.box.com/shared/static/pv2i49t4p7238gp34txhm7jcl7domw58.hdf5",
            mg_im="https://utexas.box.com/shared/static/uagjbu3y2ds0hzm7789nuj0ppxp9dxxl.hdf5",
        ),
    ),
    CoffeeServeMug=dict(
        horizon=600,
        human_path="v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01",
        mg_path="v0.1/single_stage/kitchen_coffee/CoffeeServeMug/mg/2024-05-04-22-21-50",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/tse3mkxx913pf4d4ij7fplchtv1rchkq.hdf5",
            human_im="https://utexas.box.com/shared/static/ts3537f93dzjpkux19syy0pndw5re231.hdf5",
            mg_im="https://utexas.box.com/shared/static/ikkr7qu89v9e0o1kwiqttkxllaxvv2n4.hdf5",
        ),
    ),
    CoffeePressButton=dict(
        horizon=300,
        human_path="v0.1/single_stage/kitchen_coffee/CoffeePressButton/2024-04-25",
        mg_path="v0.1/single_stage/kitchen_coffee/CoffeePressButton/mg/2024-05-04-22-21-32",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/t6oblmeilexe9ndca4ccaxdclw1tzps5.hdf5",
            human_im="https://utexas.box.com/shared/static/l5dnmcfd0r36vhdqgjchxo20vajt7ohl.hdf5",
            mg_im="https://utexas.box.com/shared/static/y5zm9mlslfg8p4jkpwnxlizcsvsca1mb.hdf5",
        ),
    ),
    TurnOnMicrowave=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/2024-04-25",
        mg_path="v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/mg/2024-05-04-22-40-00",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/o1sp7qrcd97b6jo68olx58l35wffpbna.hdf5",
            human_im="https://utexas.box.com/shared/static/t6eromogy5is2no3s1e5bk1nb2hgab2k.hdf5",
            mg_im="https://utexas.box.com/shared/static/t1n5oijudf8yn85bj8v64b1codde7kye.hdf5",
        ),
    ),
    TurnOffMicrowave=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_microwave/TurnOffMicrowave/2024-04-25",
        mg_path="v0.1/single_stage/kitchen_microwave/TurnOffMicrowave/mg/2024-05-04-22-39-23",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/m2exfytqjd0496y70b3ou2i8993ryb50.hdf5",
            human_im="https://utexas.box.com/shared/static/0drm2h7fgd5857x8xgcj1lph23srpbj1.hdf5",
            mg_im="https://utexas.box.com/shared/static/7c8bku46us9a8sddwg3zk6z3p4ce6ya0.hdf5",
        ),
    ),
    NavigateKitchen=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_navigate/NavigateKitchen/2024-05-09",
        mg_path=None,
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/pzvzuvmkla74kro84yw13uwjr2g89z8m.hdf5",
            human_im="https://utexas.box.com/shared/static/mbi6011svy2zb436wsf2hhc337aduwmw.hdf5",
        ),
    ),
)


MULTI_STAGE_TASK_DATASETS = OrderedDict(
    ArrangeVegetables=dict(
        human_path="v0.1/multi_stage/chopping_food/ArrangeVegetables/2024-05-11",
        horizon=1200,
        activity="chopping_food",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/hiwbv68h3vty9ksqsdhcw2xhuvkwcf3o.hdf5",
            human_im="https://utexas.box.com/shared/static/ovzt0bod4nrrack6dak12jn1fx2cnwbw.hdf5",
        ),
    ),
    MicrowaveThawing=dict(
        human_path="v0.1/multi_stage/defrosting_food/MicrowaveThawing/2024-05-11",
        horizon=1000,
        activity="defrosting_food",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/wreambog4jak8og9nbhm0nr2t6w5ygdt.hdf5",
            human_im="https://utexas.box.com/shared/static/ehwo6728r3qyi8g0h70zhawijojqotnm.hdf5",
        ),
    ),
    RestockPantry=dict(
        human_path="v0.1/multi_stage/restocking_supplies/RestockPantry/2024-05-10",
        horizon=1000,
        activity="restocking_supplies",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/t2g6rdyh006bw09cxxxamd7thton5grl.hdf5",
            human_im="https://utexas.box.com/shared/static/35yr2xmsmwq0xkknvlmlivh9ifste71q.hdf5",
        ),
    ),
    PreSoakPan=dict(
        human_path="v0.1/multi_stage/washing_dishes/PreSoakPan/2024-05-10",
        horizon=1500,
        activity="washing_dishes",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/d4vg2pag7465xp99vej1skk3frmohzk1.hdf5",
            human_im="https://utexas.box.com/shared/static/etotr8tn0z57ttb311bibh0mdmc65j96.hdf5",
        ),
    ),
    PrepareCoffee=dict(
        human_path="v0.1/multi_stage/brewing/PrepareCoffee/2024-05-07",
        horizon=1000,
        activity="brewing",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/qhtle2ccvm17g65ecg2vrudpkl9v5wkk.hdf5",
            human_im="https://utexas.box.com/shared/static/qq3p4ctlmiv751jx5av92aaz9dxylw76.hdf5",
        ),
    ),
)


def get_ds_path(task, ds_type, return_info=False):
    if task in SINGLE_STAGE_TASK_DATASETS:
        ds_config = SINGLE_STAGE_TASK_DATASETS[task]
    elif task in MULTI_STAGE_TASK_DATASETS:
        ds_config = MULTI_STAGE_TASK_DATASETS[task]
    else:
        raise ValueError

    if ds_type == "human_raw":
        folder = ds_config["human_path"]
        fname = "demo.hdf5"
    elif ds_type == "human_im":
        folder = ds_config["human_path"]
        if task in SINGLE_STAGE_TASK_DATASETS:
            fname = "demo_gentex_im128_randcams.hdf5"
        elif task in MULTI_STAGE_TASK_DATASETS:
            fname = "demo_im128.hdf5"
    elif ds_type == "mg_im":
        # mg dataset is not available for all tasks
        folder = ds_config.get("mg_path", None)
        fname = "demo_gentex_im128_randcams.hdf5"
    else:
        raise ValueError

    # if dataset type is not registered, return None
    if folder is None:
        ret = (None, None) if return_info is True else None
        return ret

    if macros.DATASET_BASE_PATH is None:
        ds_base_path = os.path.join(Path(robocasa.__path__[0]).parent.absolute(), "datasets")
    else:
        ds_base_path = macros.DATASET_BASE_PATH
    ds_path = os.path.join(ds_base_path, folder, fname)

    if return_info is False:
        return ds_path

    ds_info = {}
    ds_info["url"] = ds_config["download_links"][ds_type]
    ds_info["horizon"] = ds_config["horizon"]
    return ds_path, ds_info
