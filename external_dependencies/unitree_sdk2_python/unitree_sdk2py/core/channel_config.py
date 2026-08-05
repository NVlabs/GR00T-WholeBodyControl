ChannelConfigHasInterface = '''<?xml version="1.0" encoding="UTF-8" ?>
    <CycloneDDS>
        <Domain Id="any">
            <General>
                <Interfaces>
                    <NetworkInterface name="$__IF_NAME__$" priority="default" multicast="false"/>
                </Interfaces>
            </General>
            <Discovery>
                <ParticipantIndex>auto</ParticipantIndex>
            </Discovery>
        </Domain>
    </CycloneDDS>'''

ChannelConfigAutoDetermine = '''<?xml version="1.0" encoding="UTF-8" ?>
    <CycloneDDS>
        <Domain Id="any">
            <General>
                <Interfaces>
                    <NetworkInterface name="lo" priority="default" multicast="false" />
                    <NetworkInterface autodetermine="true" priority="default" multicast="false" />
                </Interfaces>
            </General>
            <Discovery>
                <ParticipantIndex>auto</ParticipantIndex>
            </Discovery>
        </Domain>
    </CycloneDDS>'''
